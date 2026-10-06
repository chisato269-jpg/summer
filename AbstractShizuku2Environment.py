"""
AbstractShizuku2Environment.py
Shizuku2の抽象環境クラス
"""

import os
import asyncio
from enum import Enum
from datetime import datetime
from szklib import VRFSystemCommunicator as vrc
from szklib import EnvironmentCommunicator as evc
from szklib import OccupantCommunicator as occ
from szklib import VentilationSystemCommunicator as vsc
from gymnasium import Env
from abc import ABC, abstractmethod
import shutil
import threading
import platform
import time # sleepのために追加

# ptyを条件付きインポート
if platform.system() != "Windows":
    try:
        import pty
    except ImportError:
        print("Warning: pty module not found, pty functionality will be disabled.")
        pty = None
else:
    pty = None


class AbstractShizuku2Environment(Env,ABC):
    """ Shizuku2の抽象環境クラス """

    ACC_RATE = 86400 * 10

    class emulator_status(Enum):
        Booting = 1
        Processing = 2
        Completed = 3
        Terminated = 4

    @property
    def CurrentDateTime(self)->datetime:
        # スレッドセーフではない可能性を考慮し、アクセスを制限
        try:
            if self.vrfCom and self.loop and self.loop.is_running():
                 return self.vrfCom.current_date_time()
        except Exception:
            pass
        return None


    def __init__(self, action_timestep:int=1800, backup_interval:int=100, eval_mode=False):
        super(AbstractShizuku2Environment, self).__init__()
        self._define_space()

        self.vrfCom, self.vscCom, self.envCom, self.occCom = None, None, None, None
        self.szkProcess, self.reader, self.writer = None, None, None
        self._final_dtime = None
        self._act_timestep = action_timestep
        self._backup_interval = backup_interval
        self._em_status = self.emulator_status.Terminated
        self._nepsd = 0
        self._eval_mode = eval_mode
        self.monitor_task = None

        self.os_type = platform.system()
        self.is_colab = ('COLAB_NOTEBOOK_FILE' in os.environ or
                         'COLAB_GPU' in os.environ or
                         'TBE_CREDS_ADDR' in os.environ)
        self.working_directory = os.path.abspath('Shizuku2')

        if self.is_colab:
            print("INFO: Colab environment detected.")
            dotnet_path = '/root/.dotnet'
            dotnet_executable = os.path.join(dotnet_path, 'dotnet')
            os.environ['PATH'] = f"{dotnet_path}:{os.environ.get('PATH', '')}"
            self.emulator_command = [dotnet_executable, 'Shizuku2.dll']
            self.use_pty = pty is not None
            if not self.use_pty: print("WARNING: pty module not available on Colab? Falling back to PIPE.")
            if not os.path.exists(dotnet_executable): print(f"FATAL WARNING: dotnet not found at {dotnet_executable}.")

        elif self.os_type == "Windows":
            print("INFO: Windows environment detected.")
            executable_path = os.path.join(self.working_directory, 'Shizuku2.exe')
            self.emulator_command = [executable_path]
            self.use_pty = False
            if not os.path.exists(executable_path): print(f"FATAL WARNING: Executable not found at {executable_path}.")

        elif self.os_type == "Linux" or self.os_type == "Darwin":
            print(f"INFO: Local {self.os_type} environment detected.")
            self.emulator_command = ['dotnet', 'Shizuku2.dll']
            self.use_pty = pty is not None
            if not self.use_pty: print(f"WARNING: pty module not available on {self.os_type}? Falling back to PIPE.")
        else:
            raise NotImplementedError(f"Unsupported OS: {self.os_type}")

        # ループとスレッドは reset 時に作成 ★
        self.loop = None
        self.thread = None


    def _start_background_loop(self):
        """バックグラウンドループとスレッドを開始するヘルパー"""
        if self.thread is not None and self.thread.is_alive():
            print("Warning: Attempted to start loop when already running.")
            return

        print("DEBUG: Starting background loop and thread...")
        self.loop = asyncio.new_event_loop()
        self._boot_event = asyncio.Event() # 新しいループに紐づくイベントを作成
        self.thread = threading.Thread(target=self._loop_thread_target, daemon=True)
        self.thread.start()
        # スレッドがループを開始するのを少し待つ
        time.sleep(0.2) # 少し長めに
        print(f"DEBUG: Background thread started (ID: {self.thread.ident}). Loop should be running.")


    def _stop_background_loop(self):
        """バックグラウンドループとスレッドを安全に停止するヘルパー"""
        if not self.thread or not self.thread.is_alive():
            print("DEBUG: Background loop/thread already stopped.")
            self.loop = None # 念のため None に
            self.thread = None
            return

        print("DEBUG: Stopping background event loop and thread...")
        if self.loop and self.loop.is_running():
            print("DEBUG: Cancelling monitor task...")
            future = asyncio.run_coroutine_threadsafe(self._cancel_monitor_task(), self.loop)
            try:
                future.result(timeout=10) # タイムアウト付きで待機
                print("DEBUG: Monitor task cancelled successfully.")
            except asyncio.CancelledError:
                 print("DEBUG: Monitor task cancellation confirmed.")
            except asyncio.TimeoutError:
                 print("Warning: Timeout waiting for monitor task cancellation.")
            except Exception as e:
                print(f"Warning: Exception during task cancellation: {e}")

            print("DEBUG: Requesting loop stop...")
            # ループが存在し、かつ閉じられていない場合のみ停止命令を送る
            if self.loop is not None and not self.loop.is_closed():
                self.loop.call_soon_threadsafe(self.loop.stop)

        print("DEBUG: Joining background thread...")
        self.thread.join(timeout=5) # タイムアウト付きで待機
        if self.thread.is_alive():
            print("Warning: Background thread did not join gracefully.")
        else:
             print("DEBUG: Background thread joined successfully.")

        self.loop = None
        self.thread = None
        print("DEBUG: Background loop and thread cleanup finished.")


    def _loop_thread_target(self):
        """バックグラウンドスレッドでイベントループを実行し続ける"""
        loop_id = id(self.loop) if self.loop else 'None'
        print(f"DEBUG: Background thread started. Setting event loop {loop_id}...")
        try:
             asyncio.set_event_loop(self.loop)
             print(f"DEBUG: Event loop {loop_id} set. Running forever...")
             self.loop.run_forever()
             # run_forever が終了した後 (loop.stop() が呼ばれた後)
             print(f"DEBUG: Event loop {loop_id} run_forever exited.")
        except Exception as e:
             print(f"ERROR in background thread target: {e}")
        finally:
            # ループ終了時に残っているタスクをキャンセル
            try:
                if self.loop and self.loop.is_running(): # まだ動いていれば
                    print(f"DEBUG: Cancelling remaining tasks in loop {loop_id}...")
                    tasks = asyncio.all_tasks(self.loop)
                    if tasks:
                        for task in tasks:
                            if not task.done(): task.cancel()
                        print(f"DEBUG: Requested cancellation for {len(tasks)} tasks.")
                    else:
                         print("DEBUG: No remaining tasks to cancel.")
            except Exception as e:
                 print(f"Error cancelling remaining tasks: {e}")
            # ループを閉じる
            finally: # ネストしたfinallyで確実に閉じる
                if self.loop and not self.loop.is_closed():
                    print(f"DEBUG: Closing event loop {loop_id}...")
                    self.loop.close()
                    print(f"DEBUG: Background event loop {loop_id} closed.")

    # region gymnasiumのEnv実装 (同期ラッパー)

    def reset(self, seed=None, options=None):
        """reset時にループとスレッドを再起動"""
        print("\n--- Resetting Environment ---")
        self._stop_background_loop() # 古いものを確実に停止
        self._start_background_loop() # 新しいものを開始
        
        if not self.loop or not self.loop.is_running():
             print("FATAL ERROR: Failed to start background loop in reset.")
             return self.observation_space.sample(), {"error": "Loop start failed"}

        future = asyncio.run_coroutine_threadsafe(self._async_reset(), self.loop)
        try:
             # reset自体のタイムアウトも考慮 (例: 90秒)
             return future.result(timeout=90)
        except asyncio.TimeoutError:
             print("FATAL ERROR: Timeout waiting for _async_reset to complete.")
             self._stop_background_loop() # ハングした可能性があるので強制停止
             return self.observation_space.sample(), {"error": "Reset timed out"}
        except Exception as e:
             print(f"FATAL ERROR during reset execution: {e}")
             self._stop_background_loop()
             return self.observation_space.sample(), {"error": f"Reset execution failed: {e}"}

    def step(self, action):
        """step時にループが存在するか確認"""
        if self.loop is None or not self.loop.is_running():
             print("ERROR: Event loop is not running in step. Forcing reset.")
             obs, info = self.reset()
             # stepはterminated/truncatedを返す必要があるので調整
             is_terminated = info.get("error", None) is not None
             is_truncated = False
             return obs, 0.0, is_terminated, is_truncated, info

        # Terminated状態はエピソード終了を示すので、resetに任せる
        if self._em_status == self.emulator_status.Terminated:
            print("DEBUG: Emulator terminated state detected in step, calling reset...")
            obs, info = self.reset()
            # resetが成功した場合は新しいエピソードの開始なのでFalseを返す
            is_terminated = info.get("error", None) is not None
            is_truncated = False
            return obs, 0.0, is_terminated, is_truncated, info # rewardは0
        
        future = asyncio.run_coroutine_threadsafe(self._async_step(action), self.loop)
        try:
             # stepにもタイムアウトを追加 (例: 60秒)
             return future.result(timeout=60)
        except asyncio.TimeoutError:
            print(f"FATAL ERROR: Timeout waiting for _async_step(action={action}) to complete.")
            self._stop_background_loop() # ハングの可能性
            # エラーを示すために terminated=True を返す
            return self.observation_space.sample(), 0.0, True, False, {"error": "Step timed out"}
        except Exception as e:
            print(f"FATAL ERROR during step execution: {e}")
            self._stop_background_loop()
            return self.observation_space.sample(), 0.0, True, False, {"error": f"Step execution failed: {e}"}

    def close(self):
        """close時もループとスレッドを停止"""
        print("Closing environment...")
        self._stop_background_loop()
        print("Environment closed.")

    # endregion

    # region 非同期ヘルパーメソッド

    async def _cancel_monitor_task(self):
        """監視タスクと関連プロセス・リソースを安全にキャンセル・終了"""
        print("DEBUG: Starting _cancel_monitor_task...")
        # (タスクキャンセル)
        if self.monitor_task and not self.monitor_task.done():
            print("DEBUG: Cancelling monitor task...")
            self.monitor_task.cancel()
            try:
                await asyncio.wait_for(self.monitor_task, timeout=5.0) # キャンセル完了待機
            except (asyncio.CancelledError, asyncio.TimeoutError):
                print("DEBUG: Monitor task cancellation processed or timed out.")
            except Exception as e:
                print(f"Warning: Exception while waiting for cancelled monitor task: {e}")
        self.monitor_task = None
        
        # (プロセスkill)
        if self.szkProcess and self.szkProcess.returncode is None:
            print("DEBUG: Killing simulator process...")
            self.szkProcess.kill()
            try:
                await asyncio.wait_for(self.szkProcess.wait(), timeout=5.0)
                print("DEBUG: Simulator process terminated.")
            except asyncio.TimeoutError:
                print("Warning: Process did not terminate gracefully after kill.")
            self.szkProcess = None
        
        # (pty/pipeストリームクローズ)
        try:
            print("DEBUG: Closing IO streams...")
            if self.reader and self.use_pty: self.reader.feed_eof()
            if self.writer:
                self.writer.close()
                await asyncio.wait_for(self.writer.wait_closed(), timeout=2.0)
            print("DEBUG: IO streams closed.")
        except (asyncio.TimeoutError, Exception) as e: print(f"Warning: Error closing streams: {e}")
        self.reader, self.writer = None, None

        # (BACnetデバイスクローズ - run_in_executor を使用)
        try:
            print("DEBUG: Closing BACnet devices via executor...")
            close_futures = []
            loop = asyncio.get_running_loop()

            if self.vrfCom and hasattr(self.vrfCom.bacdevice, 'close'):
                close_futures.append(loop.run_in_executor(None, self.vrfCom.bacdevice.close))
            if self.vscCom and hasattr(self.vscCom.bacdevice, 'close'):
                close_futures.append(loop.run_in_executor(None, self.vscCom.bacdevice.close))
            if self.envCom and hasattr(self.envCom.bacdevice, 'close'):
                close_futures.append(loop.run_in_executor(None, self.envCom.bacdevice.close))
            if self.occCom and hasattr(self.occCom.bacdevice, 'close'):
                close_futures.append(loop.run_in_executor(None, self.occCom.bacdevice.close))

            if close_futures:
                results = await asyncio.gather(*close_futures, return_exceptions=True)
                print(f"DEBUG: BACnet devices close results via executor: {results}")
            else:
                 print("DEBUG: No BACnet devices to close.")

        except Exception as e:
            print(f"Warning: Error closing BACnet devices via executor: {e}")
        finally: # 確実にNoneにする
             self.vrfCom, self.vscCom, self.envCom, self.occCom = None, None, None, None
        
        print("DEBUG: Finished _cancel_monitor_task.")


    async def _async_reset(self):
        """状態を初期化 (非同期)"""
        print("DEBUG: Starting _async_reset...")
        # _cancel_monitor_task は reset() 側で呼ばれる

        self._em_status = self.emulator_status.Booting
        self._boot_event.clear() # イベントをクリア
        print("DEBUG: Creating monitor task...")
        self.monitor_task = self.loop.create_task(self._run_shizuku2())

        print("DEBUG: Waiting for boot event...")
        try:
            await asyncio.wait_for(self._boot_event.wait(), timeout=60.0) # 起動待機
            print("DEBUG: Boot event received.")
        except asyncio.TimeoutError:
            print("FATAL ERROR: Simulator boot timed out.")
            self._em_status = self.emulator_status.Terminated
            await self._cancel_monitor_task() # タイムアウト時もクリーンアップ

        if self._em_status == self.emulator_status.Terminated:
             print("Error: Simulator failed to boot or timed out. Returning a dummy observation.")
             return self.observation_space.sample(), {"error": "Simulator boot failed"}

        print("DEBUG: Observing initial state...")
        obs = await self._observe_state()
        print("DEBUG: Finished _async_reset.")
        return obs, {}


    async def _async_step(self, action):
        """状態更新処理 (非同期)"""
        # print(f"DEBUG: Starting _async_step with action {action}") # アクションが大きい場合ログが冗長になる
        await self._perform_action(action)
        last_dt = self.CurrentDateTime
        await self.vrfCom.change_pause_timer(self._act_timestep)
        
        # print("DEBUG: Waiting for simulator pause...")
        pause_start_time = self.loop.time()
        while not self.vrfCom.is_paused or (last_dt == self.CurrentDateTime):
            await asyncio.sleep(0.01)
            if self.loop.time() - pause_start_time > 30.0:
                 print("Warning: Timed out waiting for simulator pause. Proceeding anyway.")
                 break
        # print("DEBUG: Simulator paused.")

        current_sim_time_str = self.CurrentDateTime.strftime('%Y/%m/%d %H:%M:%S') if self.CurrentDateTime else "N/A"
        print(f"{current_sim_time_str}")

        is_final_step = self._final_dtime <= self.CurrentDateTime if self._final_dtime and self.CurrentDateTime else False
        rwd = await self._calculate_reward(is_final_step)
        obs = await self._observe_state()

        if is_final_step:
            print("DEBUG: Final step detected.")
            await self.vrfCom.change_pause_timer(86400 * 10)
            self._em_status = self.emulator_status.Completed
            
            print("DEBUG: Waiting for monitor task to finish naturally...")
            if self.monitor_task:
                try:
                    await asyncio.wait_for(self.monitor_task, timeout=10.0)
                    print("DEBUG: Monitor task finished.")
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    print("Warning: Monitor task did not finish cleanly after episode end.")
            
            if self._nepsd % self._backup_interval == 0:
                print(f"DEBUG: Backing up data for episode {self._nepsd}...")
                move_dir = os.path.join("./szkdata_back", ("szkdata_eval" if self._eval_mode else f"szkdata_{self._nepsd}"))
                if os.path.exists("./Shizuku2/data"):
                    try:
                        shutil.copytree("./Shizuku2/data", move_dir, dirs_exist_ok=True)
                        print(f"DEBUG: Data backed up to {move_dir}")
                    except Exception as e:
                        print(f"Warning: Failed to backup data: {e}")
            self._nepsd += 1
            print("DEBUG: Finished final step processing.")

        # print("DEBUG: Finished _async_step.")
        return obs, rwd, is_final_step, False, {}


    # 標準エラー出力を読み捨てるヘルパー（Windows用）
    async def _drain_stderr(self, stderr_reader):
        """ stderrを非同期で読み捨て、パイプが詰まるのを防ぐ (主にWindows用) """
        try:
            while True:
                line = await stderr_reader.readline()
                if not line:
                    break
                print(f"[Shizuku2 stderr] {line.decode('utf-8', errors='ignore').strip()}")
        except Exception:
            pass # プロセス終了などでリーダーが閉じられた場合

    
    async def _run_shizuku2(self):
        """シミュレータの準備と監視を行うメインコルーチン"""
        process_started_successfully = False
        loop = asyncio.get_running_loop() # このコルーチンが動くループを取得
        print(f"DEBUG: _run_shizuku2 started in loop {id(loop)}.")
        try:
            command = self.emulator_command
            working_directory = self.working_directory
            current_env = os.environ.copy()

            if self.use_pty:
                master, slave = pty.openpty()
                self.szkProcess = await asyncio.create_subprocess_exec(
                    *command, cwd=working_directory,
                    stdin=slave, stdout=slave, stderr=slave, env=current_env
                )
                os.close(slave)
                process_started_successfully = True

                self.reader = asyncio.StreamReader(loop=self.loop)
                protocol = asyncio.StreamReaderProtocol(self.reader)
                await self.loop.connect_read_pipe(lambda: protocol, os.fdopen(master, 'rb'))

                writer_transport, writer_protocol = await self.loop.connect_write_pipe(
                    asyncio.streams.FlowControlMixin, os.fdopen(master, 'wb')
                )
                self.writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, self.loop)
            
            else: # Windows
                self.szkProcess = await asyncio.create_subprocess_exec(
                    *command, cwd=working_directory,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE, env=current_env
                )
                process_started_successfully = True
                self.reader = self.szkProcess.stdout
                self.writer = self.szkProcess.stdin
                # stderr読み捨てタスクを現在のループで開始
                loop.create_task(self._drain_stderr(self.szkProcess.stderr))

            print(f"DEBUG: Simulator process started (PID: {self.szkProcess.pid}).")

            if self.vrfCom is None: self.vrfCom = vrc(id=12)
            if self.vscCom is None: self.vscCom = vsc(id=13)
            if self.envCom is None: self.envCom = evc(id=14)
            if self.occCom is None: self.occCom = occ(id=15)
            print("DEBUG: Communicator objects created/reused.")

            while self.szkProcess.returncode is None:
                try:
                    line_bytes = await asyncio.wait_for(self.reader.readline(), timeout=1.0)
                    if not line_bytes:
                         print("DEBUG: EOF received from simulator stdout.")
                         break
                    so = line_bytes.decode('utf-8', errors='ignore').strip()
                    if so: print(f"[Shizuku2] {so}")

                    if 'Press "Enter" key to continue.' in so:
                        print("[Shizuku2] 'Press Enter' detected. Stabilizing...")
                        await asyncio.sleep(2)
                        print("DEBUG: Subscribing to COV...")
                        await self.vrfCom.subscribe_date_time_cov()
                        print("DEBUG: Syncing date/time...")
                        while not await self.vrfCom.sync_date_time(): await asyncio.sleep(0.1)
                        print("DEBUG: Setting pause timer...")
                        while not await self.vrfCom.change_pause_timer(0): await asyncio.sleep(0.1)
                        print("DEBUG: Getting end date/time...")
                        while not (result := await self.vrfCom.get_end_dateTime())[0]: await asyncio.sleep(0.1)
                        self._final_dtime = result[1]
                        print(f"DEBUG: End date/time set to {self._final_dtime}")
                        print("DEBUG: Sending Enter to simulator...")
                        self.writer.write(b"\r\n")
                        await self.writer.drain()
                        print("DEBUG: Setting acceleration rate...")
                        while not await self.vrfCom.change_acceleration_rate(self.ACC_RATE): await asyncio.sleep(0.1)
                        self._em_status = self.emulator_status.Processing
                        print("DEBUG: Setting boot event (success)...")
                        self._boot_event.set() # 正常起動時にイベントをセット
                        print("DEBUG: Simulator is now processing.")

                    if 'Emulation finished.' in so:
                        print("DEBUG: 'Emulation finished' detected.")
                        while self._em_status != self.emulator_status.Completed:
                            await asyncio.sleep(0.1)
                        print("DEBUG: Sending Enter to exit simulator...")
                        self.writer.write(b"\r\n")
                        await self.writer.drain()
                        try:
                             print("DEBUG: Unsubscribing COV...")
                             self.vrfCom.unsubscribe_date_time_cov()
                        except Exception as unsub_e: print(f"Warning: Error unsubscribing COV: {unsub_e}")
                        break # 監視ループを正常に抜ける
                except asyncio.TimeoutError:
                    continue # タイムアウトは正常、ループを続ける
        
        except asyncio.CancelledError:
            print("Shizuku2 monitoring task was cancelled.")
            self._em_status = self.emulator_status.Terminated

        except FileNotFoundError:
            print(f"FATAL ERROR: Command '{command[0]}' not found.")
            # ... (エラーメッセージ) ...
            self.szkProcess = None 
            self._em_status = self.emulator_status.Terminated
            if not self._boot_event.is_set(): self._boot_event.set() # 起動失敗時はreset()を解除

        except Exception as e:
            print(f"An unexpected error occurred in _run_shizuku2: {e}")
            self._em_status = self.emulator_status.Terminated
            if not process_started_successfully and not self._boot_event.is_set():
                 self._boot_event.set() # 起動失敗時はreset()を解除

        finally:
            print("Shizuku2 monitoring coroutine has ended.")
            # タスク終了時に最終状態を設定し、boot_eventを確実にセット
            # (ただし、キャンセルされた場合は除く - キャンセルは外部からの要求)
            current_task = asyncio.current_task(loop=loop) if loop else None
            is_cancelled = current_task.cancelled() if current_task else False

            if not is_cancelled:
                 self._em_status = self.emulator_status.Terminated
                 if not self._boot_event.is_set():
                      print("DEBUG: Setting boot event in finally (task not cancelled)...")
                      self._boot_event.set()
                 
            # クリーンアップ漏れを防ぐため、念のためプロセスをkill (既に終了していれば何もしない)
            if self.szkProcess and self.szkProcess.returncode is None:
                print("DEBUG: Killing process in finally block just in case...")
                self.szkProcess.kill()
                # ここで await self.szkProcess.wait() するとデッドロックの可能性
            
            print(f"DEBUG: Final emulator status: {self._em_status}")
            print(f"DEBUG: Boot event status: {self._boot_event.is_set()}")
                
    # endregion

    # region 抽象メソッド (変更なし)
    @abstractmethod
    def _define_space(self): pass

    @abstractmethod
    async def _perform_action(self, action): pass

    @abstractmethod
    async def _observe_state(self): pass

    @abstractmethod
    async def _calculate_reward(self, is_final_step): pass

    # endregion