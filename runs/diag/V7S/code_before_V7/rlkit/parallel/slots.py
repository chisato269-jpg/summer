"""rlkit.parallel.slots — 組の定義、複製の準備と照合、ipadd の書き換え、組のロック、
使用中ポートの確認（指示書PL2 第1節・第6-2節）。

組 k（0〜7）はアドレス 127.0.0.(k+1) を ipadd・device_ip・emulator_ip に使い、
通信IDは現行の4つ（rlkit/comm.py の CLIENT_DEVICE_IDS）に 10k を足したものを使う。
エミュレータは原本と同じ階層の「<原本名>_slot{k}」に置いた複製から起動する。
組を指定しない従来の走行はこのモジュールを通らない。
"""
import atexit
import dataclasses
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess

from rlkit import comm
from rlkit.parallel import procs

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORIGINAL_DIR = os.path.join(_REPO_ROOT, "Shizuku2")

N_SLOTS = 8
SLOT_ID_STEP = 10
CLIENT_KEYS = ("vrf", "vent", "env", "occ")
# クライアントの待ち受けポート = 0xBAC0 + id（szklib/PresentValueReadWriter.py:42、PL1のC-2で確認）
CLIENT_PORT_BASE = 0xBAC0
# エミュレータ側の装置のポート。組によらず同じで、アドレスによって区別される。
EMULATOR_PORTS = tuple(range(47809, 47818))
DATETIME_CONTROLLER_PORT = 47809
# エミュレータ側の装置の Device ID（reference.pdf 4.2節。本書第1節「禁止する値」）
FORBIDDEN_IDS = frozenset({1, 2, 3, 4, 5, 6, 9})
EMULATOR_EXE_NAME = "Shizuku2.exe"
# PL1-6で「エミュレータが書き出す」と分類した8ファイル。複製の照合から除く。
EMULATOR_OUTPUT_RELPATHS = frozenset(os.path.join("data", n) for n in (
    "general.csv", "occupant.csv", "vent.csv", "vrf.csv", "zone.csv",
    "dissatisfaction.csv", "result.txt", "result.szk"))
LOCK_NAME = "slot.lock"
MANIFEST_NAME = "slot_manifest.json"
# 組の管理のために複製の最上位に置くファイル（原本には無い）。照合から除く。
SLOT_ADMIN_NAMES = frozenset({LOCK_NAME, MANIFEST_NAME})

_BOM = b"\xef\xbb\xbf"
_IPADD_KEY = b"ipadd="
_IPV4_RE = re.compile(r"^([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})$")


class SlotError(RuntimeError):
    pass


def _now_iso():
    return datetime.datetime.now().isoformat()


def _write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =====================================================================
# 組の仕様
# =====================================================================

@dataclasses.dataclass(frozen=True)
class SlotSpec:
    index: int
    address: str
    client_id_items: tuple  # (("vrf", 12), ("vent", 13), ("env", 14), ("occ", 15)) の形
    replica_dir: str

    @property
    def client_ids(self):
        return dict(self.client_id_items)

    @property
    def device_ip(self):
        return self.address

    @property
    def emulator_ip(self):
        return self.address

    @property
    def client_ports(self):
        ids = self.client_ids
        return tuple(CLIENT_PORT_BASE + ids[k] for k in CLIENT_KEYS)

    @property
    def datetime_addr(self):
        return f"{self.address}:{DATETIME_CONTROLLER_PORT}"

    @property
    def executable(self):
        return os.path.join(self.replica_dir, EMULATOR_EXE_NAME)

    @property
    def lock_path(self):
        return os.path.join(self.replica_dir, LOCK_NAME)

    @property
    def data_dir(self):
        return os.path.join(self.replica_dir, "data")

    @property
    def setting_ini(self):
        return os.path.join(self.replica_dir, "setting.ini")

    def to_record(self):
        return {
            "index": self.index, "address": self.address,
            "device_ip": self.device_ip, "emulator_ip": self.emulator_ip,
            "client_ids": self.client_ids, "client_ports": list(self.client_ports),
            "datetime_addr": self.datetime_addr, "replica_dir": self.replica_dir,
            "lock_path": self.lock_path,
        }


def replica_dir_for(k, original_dir=ORIGINAL_DIR):
    parent, name = os.path.split(os.path.normpath(original_dir))
    return os.path.join(parent, f"{name}_slot{k}")


def make_slot(k, original_dir=ORIGINAL_DIR, base_ids=None):
    """組 k の仕様を作り、値を検査して返す。k は 0〜7 の整数に限る。"""
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < N_SLOTS:
        raise SlotError(f"組番号は0〜7の整数に限る: {k!r}")
    base = dict(base_ids if base_ids is not None else comm.CLIENT_DEVICE_IDS)
    ids = tuple((key, base[key] + SLOT_ID_STEP * k) for key in CLIENT_KEYS)
    spec = SlotSpec(index=k, address=f"127.0.0.{k + 1}", client_id_items=ids,
                    replica_dir=replica_dir_for(k, original_dir))
    validate_slots([spec])
    return spec


def parse_slot_arg(value, original_dir=ORIGINAL_DIR):
    """実行の入口の --slot の値を組の仕様へ変える。None は「指定なし」。"""
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        if not re.fullmatch(r"[0-9]+", v):
            raise SlotError(f"組番号は0〜7の整数に限る: {value!r}")
        value = int(v)
    return make_slot(value, original_dir)


def validate_slots(slots):
    """第1節「禁止する値」（IDの1〜6と9、ポートの47809〜47817）と、組の間の重複を誤りとする。"""
    errors = []
    seen = {"index": {}, "address": {}, "replica": {}, "id": {}, "port": {}}

    def _dup(kind, value, owner):
        if value in seen[kind]:
            errors.append(f"{kind} {value!r} が組{seen[kind][value]}と組{owner}で重複")
        else:
            seen[kind][value] = owner

    for s in slots:
        if isinstance(s.index, bool) or not isinstance(s.index, int) or not 0 <= s.index < N_SLOTS:
            errors.append(f"組番号は0〜7の整数に限る: {s.index!r}")
        ids = s.client_ids
        if set(ids) != set(CLIENT_KEYS):
            errors.append(f"組{s.index}の通信IDの項目が {sorted(ids)}（{list(CLIENT_KEYS)} であるべき）")
            continue
        for key in CLIENT_KEYS:
            i = ids[key]
            port = CLIENT_PORT_BASE + i
            if i in FORBIDDEN_IDS:
                errors.append(f"組{s.index}の{key}の通信ID {i} は禁止値（1〜6と9）")
            if port in EMULATOR_PORTS:
                errors.append(f"組{s.index}の{key}のポート {port} はエミュレータ側の装置のポート（47809〜47817）")
            _dup("id", i, s.index)
            _dup("port", port, s.index)
        _dup("index", s.index, s.index)
        _dup("address", s.address, s.index)
        _dup("replica", procs.norm_path(s.replica_dir), s.index)
    if errors:
        raise SlotError("; ".join(errors))


# =====================================================================
# setting.ini の ipadd の書き換え
# =====================================================================

def _check_ipv4(ip):
    m = _IPV4_RE.match(ip or "")
    if not m or any(int(g) > 255 for g in m.groups()):
        raise SlotError(f"IPv4アドレスの形式ではない: {ip!r}")


def find_ipadd_line(data):
    """(行のリスト, ipadd の行の番号) を返す。ipadd の行が無い、または2行以上あれば誤り。"""
    lines = data.split(b"\n")
    hits = []
    for i, ln in enumerate(lines):
        body = ln[len(_BOM):] if (i == 0 and ln.startswith(_BOM)) else ln
        if body.lstrip(b" \t").startswith(_IPADD_KEY):
            hits.append(i)
    if len(hits) != 1:
        raise SlotError(f"setting.ini の ipadd の行が {len(hits)} 行ある（1行であるべき。追記はしない）")
    return lines, hits[0]


def _ipadd_value_span(line):
    start = line.index(_IPADD_KEY) + len(_IPADD_KEY)
    semi = line.find(b";", start)
    if semi < 0:
        raise SlotError("ipadd の行に ';' が無い（書式 ipadd=値; を想定）")
    return start, semi


def rewrite_ipadd_bytes(data, new_ip):
    """ipadd の値だけをバイト単位で置き換える。BOM・改行コード・他の行・行内の書式
    （'ipadd=' と ';' 以降のコメント）はそのまま保つ。"""
    _check_ipv4(new_ip)
    lines, idx = find_ipadd_line(data)
    line = lines[idx]
    start, semi = _ipadd_value_span(line)
    lines[idx] = line[:start] + new_ip.encode("ascii") + line[semi:]
    return b"\n".join(lines)


def read_ipadd_value(data):
    lines, idx = find_ipadd_line(data)
    start, semi = _ipadd_value_span(lines[idx])
    return lines[idx][start:semi].decode("ascii", errors="replace")


# =====================================================================
# 複製の準備と照合
# =====================================================================

def tree_hashes(root, skip_relpaths=frozenset(), skip_top_names=frozenset()):
    """root 以下の全ファイルの SHA-256 を {相対パス: ハッシュ} で返す。"""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            if rel in skip_relpaths:
                continue
            if os.sep not in rel and rel in skip_top_names:
                continue
            out[rel] = sha256_file(full)
    return out


def prepare_replica(slot, original_dir=ORIGINAL_DIR):
    """複製が無ければ原本を丸ごと複製して ipadd だけを書き換える。複製があれば照合だけを行う。"""
    created = False
    if not os.path.isdir(slot.replica_dir):
        partial = slot.replica_dir + ".partial"
        if os.path.exists(partial):
            raise SlotError(f"前回の複製が途中で止まった跡がある（中身を確かめてから削除すること）: {partial}")
        shutil.copytree(original_dir, partial)
        ini = os.path.join(partial, "setting.ini")
        with open(ini, "rb") as f:
            data = f.read()
        with open(ini, "wb") as f:
            f.write(rewrite_ipadd_bytes(data, slot.address))
        os.rename(partial, slot.replica_dir)
        _write_json_atomic(os.path.join(slot.replica_dir, MANIFEST_NAME), {
            "slot": slot.index, "original_dir": os.path.abspath(original_dir), "created_at": _now_iso(),
            "original_hashes": tree_hashes(original_dir, EMULATOR_OUTPUT_RELPATHS),
        })
        created = True
    result = verify_replica(slot, original_dir)
    result["created_now"] = created
    return result


def verify_replica(slot, original_dir=ORIGINAL_DIR):
    """原本と複製の全ファイルの SHA-256 を比べ、差が setting.ini の ipadd の値だけであることを
    確かめる。原本側が複製の作成時（前回の照合時）から変わっていれば誤りとする。"""
    manifest_path = os.path.join(slot.replica_dir, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        raise SlotError(f"照合記録 {MANIFEST_NAME} が無い（この機構で作った複製ではない）: {slot.replica_dir}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    errors = []
    orig = tree_hashes(original_dir, EMULATOR_OUTPUT_RELPATHS)
    recorded = manifest.get("original_hashes") or {}
    if recorded != orig:
        changed = sorted(k for k in set(recorded) | set(orig) if recorded.get(k) != orig.get(k))
        errors.append(f"原本が前回の照合時から変わっている: {changed[:10]}")

    repl = tree_hashes(slot.replica_dir, EMULATOR_OUTPUT_RELPATHS, SLOT_ADMIN_NAMES)
    missing = sorted(set(orig) - set(repl))
    extra = sorted(set(repl) - set(orig))
    differing = sorted(k for k in set(orig) & set(repl) if orig[k] != repl[k])
    if missing:
        errors.append(f"複製に無いファイル: {missing[:10]}")
    if extra:
        errors.append(f"原本に無いファイル: {extra[:10]}")
    unexpected = [k for k in differing if k != "setting.ini"]
    if unexpected:
        errors.append(f"setting.ini 以外に中身の違うファイル: {unexpected[:10]}")

    with open(os.path.join(original_dir, "setting.ini"), "rb") as f:
        orig_ini = f.read()
    with open(slot.setting_ini, "rb") as f:
        repl_ini = f.read()
    if repl_ini != rewrite_ipadd_bytes(orig_ini, slot.address):
        errors.append("複製の setting.ini が「原本の ipadd の値だけを組のアドレスに置き換えたもの」と一致しない")
    ol, rl = orig_ini.split(b"\n"), repl_ini.split(b"\n")
    diff_lines = [i for i, (a, b) in enumerate(zip(ol, rl)) if a != b]
    if len(ol) != len(rl):
        errors.append(f"setting.ini の行数が違う（原本{len(ol)}行、複製{len(rl)}行）")

    if errors:
        raise SlotError(f"組{slot.index}の複製の照合に失敗: " + "; ".join(errors))
    return {
        "slot": slot.index, "replica_dir": slot.replica_dir,
        "n_files_compared": len(orig), "setting_ini_diff_line_indices": diff_lines,
        "ipadd_value": read_ipadd_value(repl_ini),
        "excluded_from_comparison": sorted(EMULATOR_OUTPUT_RELPATHS) + sorted(SLOT_ADMIN_NAMES),
    }


# =====================================================================
# 使用中ポートの確認
# =====================================================================

_NETSTAT_UDP_RE = re.compile(r"^\s*UDP\s+(\S+)\s+\S+\s+(\d+)\s*$")


def run_netstat_udp():
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "UDP"], capture_output=True, text=True,
                             timeout=30, errors="replace")
    except Exception as exc:
        raise SlotError(f"netstat を実行できない: {exc!r}")
    if out.returncode != 0:
        raise SlotError(f"netstat が失敗した（終了コード {out.returncode}）")
    return out.stdout


def parse_netstat_udp(text):
    """netstat -ano -p UDP の出力から (アドレス, ポート, PID) の一覧を返す。"""
    rows = []
    for line in text.splitlines():
        m = _NETSTAT_UDP_RE.match(line)
        if not m:
            continue
        local, pid = m.group(1), int(m.group(2))
        if local.startswith("["):
            host, _, port = local[1:].partition("]:")
        else:
            host, _, port = local.rpartition(":")
        try:
            rows.append((host, int(port), pid))
        except ValueError:
            continue
    return rows


def find_busy_ports(slot, netstat_text=None):
    """その組のアドレス（と全アドレス 0.0.0.0）で、エミュレータ側のポートとクライアントの
    ポートを使っているプロセスを返す。0.0.0.0 は衝突しうるため保守的に含める。"""
    text = run_netstat_udp() if netstat_text is None else netstat_text
    ports = set(EMULATOR_PORTS) | set(slot.client_ports)
    return [{"address": h, "port": p, "pid": pid} for h, p, pid in parse_netstat_udp(text)
            if p in ports and h in (slot.address, "0.0.0.0")]


def assert_ports_free(slot, netstat_text=None):
    busy = find_busy_ports(slot, netstat_text)
    if busy:
        raise SlotError(f"組{slot.index}（{slot.address}）のポートを使っているプロセスがあるため起動しない: {busy}")
    return busy


# =====================================================================
# 組のロック
# =====================================================================

class SlotLock:
    """複製の中の slot.lock に、学習側（またはクライアント側）の PID と、起動したエミュレータの
    PID を書く。PID の再利用に備えて、それぞれの生成時刻も書く。"""

    def __init__(self, slot, proc_api=None):
        self.slot = slot
        self.path = slot.lock_path
        self.api = proc_api or procs.DEFAULT_API

    def read(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError) as exc:
            raise SlotError(f"ロックファイルを読めない: {self.path}: {exc!r}")

    def _write(self, rec):
        _write_json_atomic(self.path, rec)

    def acquire(self, learner_pid=None):
        """ロックを取る。書かれた学習側の PID が生きていれば起動を拒否する。学習側が死んでいて、
        書かれたエミュレータが生きており、その実行ファイルがこの組の複製の中にある場合に限り、
        そのエミュレータを PID で終了させてから取る。戻り値: 何をしたかの記録。"""
        learner_pid = os.getpid() if learner_pid is None else learner_pid
        learner_ct = self.api.create_time(learner_pid)
        report = {"slot": self.slot.index, "time": _now_iso(), "previous": None, "action": "new"}
        rec = self.read()
        if rec is not None:
            report["previous"] = rec
            lp, lct = rec.get("learner_pid"), rec.get("learner_create_time")
            if lp == learner_pid and procs._create_time_matches(learner_ct, lct):
                report["action"] = "already_own"
                return report
            if self.api.is_alive(lp, lct):
                raise SlotError(f"組{self.slot.index}はロック中（学習側 PID {lp} が生きている）。起動を拒否する")
            ep, ect = rec.get("emulator_pid"), rec.get("emulator_create_time")
            if ep is not None and self.api.is_alive(ep, ect):
                exe = self.api.exe(ep)
                if exe is not None and procs.path_is_inside(exe, self.slot.replica_dir):
                    if not self.api.kill(ep, ect):
                        raise SlotError(f"組{self.slot.index}に残っていたエミュレータ PID {ep} を終了できない")
                    report.update(action="took_over_after_killing_stale_emulator", killed_pid=ep, killed_exe=exe)
                else:
                    report.update(action="took_over_stale_lock_emulator_not_in_replica",
                                  ignored_pid=ep, ignored_exe=exe)
            else:
                report["action"] = "took_over_stale_lock"
        self._write({
            "slot": self.slot.index, "learner_pid": learner_pid, "learner_create_time": learner_ct,
            "emulator_pid": None, "emulator_create_time": None, "emulator_exe": None,
            "acquired_at": _now_iso(),
        })
        return report

    def _update_own(self, **fields):
        rec = self.read()
        if rec is None or rec.get("learner_pid") != os.getpid():
            raise SlotError(f"組{self.slot.index}のロックを自分が持っていない: {rec}")
        rec.update(fields)
        self._write(rec)

    def set_emulator(self, pid):
        self._update_own(emulator_pid=pid, emulator_create_time=self.api.create_time(pid),
                         emulator_exe=self.api.exe(pid), emulator_started_at=_now_iso())

    def clear_emulator(self):
        self._update_own(emulator_pid=None, emulator_create_time=None, emulator_exe=None)

    def release(self):
        rec = self.read()
        if rec is not None and rec.get("learner_pid") == os.getpid():
            os.remove(self.path)
            return True
        return False

    def release_if_owner_dead(self, owner_pid):
        """起動機構用：書かれた学習側の PID が owner_pid で、それがもう生きていなければ外す。"""
        rec = self.read()
        if rec is None or rec.get("learner_pid") != owner_pid:
            return False
        if self.api.is_alive(owner_pid, rec.get("learner_create_time")):
            return False
        os.remove(self.path)
        return True


# =====================================================================
# プロセス内の実行時状態
# =====================================================================

class SlotRuntime:
    """1つのプロセスの中で、ある組を使う走行の実行時状態を持つ。エピソードごとに作り直される
    EpisodeDriver の間で共有する（ロックは走行の最初の起動で取り、プロセスの終了時に外す）。"""

    def __init__(self, slot, proc_api=None, netstat_fn=None):
        self.slot = slot
        self.api = proc_api or procs.DEFAULT_API
        self.lock = SlotLock(slot, self.api)
        self._netstat_fn = netstat_fn
        self.acquired = False
        self.acquire_report = None
        self.launch_count = 0
        self.last_emulator_pid = None
        self.last_emulator_create_time = None
        self.stdout_lines = 0
        self.delayed_lines = 0

    def before_launch(self):
        if not self.acquired:
            self.acquire_report = self.lock.acquire()
            self.acquired = True
            atexit.register(self.release)
        assert_ports_free(self.slot, self._netstat_fn() if self._netstat_fn else None)

    def after_launch(self, pid):
        self.launch_count += 1
        self.last_emulator_pid = pid
        self.last_emulator_create_time = self.api.create_time(pid)
        self.lock.set_emulator(pid)

    def after_exit(self, pid, exited):
        if not exited and self.api.is_alive(pid, self.last_emulator_create_time):
            raise SlotError(f"組{self.slot.index}のエミュレータ PID {pid} を PID で終了できない（H-7）")
        self.lock.clear_emulator()

    def note_stdout_line(self, line):
        self.stdout_lines += 1
        if b"DELAYED" in line.upper():
            self.delayed_lines += 1

    def release(self):
        if self.acquired:
            self.lock.release()
            self.acquired = False

    def stats(self):
        return {"slot": self.slot.index, "launch_count": self.launch_count,
                "last_emulator_pid": self.last_emulator_pid,
                "emulator_stdout_lines": self.stdout_lines, "emulator_delayed_lines": self.delayed_lines,
                "lock_acquire_report": self.acquire_report}


_RUNTIMES = {}


def runtime_for(slot):
    rt = _RUNTIMES.get(slot.index)
    if rt is None or rt.slot != slot:
        rt = SlotRuntime(slot)
        _RUNTIMES[slot.index] = rt
    return rt
