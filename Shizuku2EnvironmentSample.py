"""
Shizuku2EnvironmentSample.py
Shizuku2 Environment Class Sample
The most complex sample that allows all control items to be changed individually for all VRFs.
"""

import numpy as np
import csv
from scipy.stats import norm
from gymnasium.spaces import Box, MultiDiscrete
from AbstractShizuku2Environment import AbstractShizuku2Environment
from szklib import VRFSystemCommunicator as vrc
from szklib import OccupantCommunicator as occ
from szklib import VentilationSystemCommunicator as vsc

IUNIT_NUM = 5+4+5+4 # 室内機の数

class Shizuku2EnvironmentSample(AbstractShizuku2Environment):
    """ Shizuku2の環境クラスサンプル """
    
    # region 初期化処理

    def __init__(self, action_timestep:int=1800, backup_interval:int=100, eval_mode=False):
        super(Shizuku2EnvironmentSample, self).__init__(action_timestep, backup_interval, eval_mode)

        # インスタンス変数の定義
        self._prevRtn = [] # 1タイムステップ前の出力

        # 成績書き出し用の空のcsvファイルを用意
        open('eci.csv', 'w').close()

    # endregion

    # region gymnasiumのEnv実装

    def reset(self, seed=None, options=None):
        self.energy = 0
        self.disrate = 0
        self.ttl_rwd = 0
        self.prev_rwd = 0
        return super().reset(seed, options)
    

    def step(self, action):
        return super().step(action)


    def render(self, mode='human'):
        super().render(mode)

    # endregion

    # region 行動・観測・報酬の定義

    def _define_space(self):
        """
        行動空間と状態空間を定義する
        """
        # 行動空間の定義
        md = []
        i_unit_num = [5,4,5,4] # 各系統の室内機の台数
        # 室外機の制御項目
        for i in range(len(i_unit_num)):
            md.extend([
                3,  # モード 0:heating, 1:cooling, 2:fan
                11, # 蒸発温度 y=x+3
                11 # 凝縮温度 y=x+40
                ])
            # 室内機の制御項目
            for _ in range(i_unit_num[i]):
                md.extend([
                    2,  # 発停 0:off, 1:on
                    31, # 室温設定値 y=22+0.2x
                    3,  # ファン風量 0:Low, 1:Middle, 2:High
                    5,  # ファン風向 0:Horizontal, 1:22.5deg, 2:45.0deg, 3:67.5deg, 4:Vertical
                    2,  # 執務者リモコン操作許可 0:Disabled, 1:Enabled
                    2,  # 全熱交換器発停 0:off, 1:on
                    2,  # 全熱交換器バイパス制御 0:Disabled, 1:Enabled
                    3   # 全熱交換器ファン風量 0:Low, 1:Middle, 2:High
                    ])
        
        self.action_space = MultiDiscrete(md)

        # 状態空間の定義（曜日+時刻+外気温度+外気湿度+ゾーン数x(ゾーン温度+ゾーン執務者数)）
        self.observation_space = Box(low=0.0, high=1.0, shape=(4 + 2 * IUNIT_NUM,), dtype=float)
        self.state = np.zeros(4 + 2 * IUNIT_NUM, dtype=np.float64)  # 初期状態はすべて0
        self.goal_state = None  # 目標状態は無し


    async def _perform_action(self, action):
        """
        行動をエミュレータに反映する
        Args:
            action: 行動
        Returns:
            None
        """

        # 室外機の制御項目
        indx:int = 0
        i_unit_num = [5,4,5,4] # 各系統の室内機の台数
        for i in range(len(i_unit_num)):
            match action[indx]:
                case 0: mode = vrc.Mode.Heating
                case 1: mode = vrc.Mode.Cooling
                case _: mode = vrc.Mode.ThermoOff
            indx+=1
            await self.vrfCom.change_evaporating_temperature(i+1,3 + action[indx])
            indx+=1
            await self.vrfCom.change_condensing_temperature(i+1,40 + action[indx])
            indx+=1
            for j in range(i_unit_num[i]):
                is_on = action[indx] == 1
                indx+=1

                # VRF室内機操作
                if is_on:
                    # 制御内容を取得
                    sp_tmp = 22 + action[indx] * 0.2
                    indx+=1
                    match action[indx]:
                        case 0: fan_spd = vrc.FanSpeed.Low
                        case 1: fan_spd = vrc.FanSpeed.Middle
                        case _: fan_spd = vrc.FanSpeed.High
                    indx+=1
                    match action[indx]:
                        case 0: fan_dir = vrc.Direction.Horizontal
                        case 1: fan_dir = vrc.Direction.Degree_225
                        case 2: fan_dir = vrc.Direction.Degree_450
                        case 3: fan_dir = vrc.Direction.Degree_675
                        case _: fan_dir = vrc.Direction.Vertical
                    indx+=1
                    usr_pmt = action[indx] == 1
                    indx+=1

                    # 制御を反映
                    await self.vrfCom.turn_on(i+1,j+1) # 空調運転
                    await self.vrfCom.change_mode(i+1,j+1,mode)
                    await self.vrfCom.change_fan_speed(i+1,j+1,fan_spd)
                    await self.vrfCom.change_direction(i+1,j+1,fan_dir)
                    if(usr_pmt):
                        await self.vrfCom.permit_local_control(i+1,j+1)
                    else:
                        await self.vrfCom.prohibit_local_control(i+1,j+1)
                        await self.vrfCom.change_setpoint_temperature(i+1,j+1,sp_tmp)                    
                else:
                    indx+=4
                    await self.vrfCom.turn_off(i+1,j+1) # 空調停止

                # 全熱交操作
                hex_on = action[indx] == 1
                indx+=1
                if hex_on:
                    # 制御内容を取得
                    hex_byps = action[indx] == 1
                    indx+=1
                    match action[indx]:
                        case 0: hex_spd = vsc.FanSpeed.Low
                        case 1: hex_spd = vsc.FanSpeed.Middle
                        case _: hex_spd = vsc.FanSpeed.High
                    indx+=1

                    # 制御を反映
                    await self.vscCom.start_ventilation(i+1,j+1)
                    await self.vscCom.change_fan_speed(i+1,j+1,hex_spd)
                    if hex_byps:
                        await self.vscCom.enable_bypass_control(i+1,j+1)
                    else:
                        await self.vscCom.disable_bypass_control(i+1,j+1)
                else:
                    indx+=2
                    await self.vscCom.stop_ventilation(i+1,j+1)


    async def _observe_state(self):
        """
        状態を観察する
        Args:
            None
        Returns:
            観察結果
        """
        indx:int = 0
        rtnList = []

        # 1: 曜日
        rtnList.append(self.CurrentDateTime.weekday() / 6)
        indx += 1

        # 2: 時刻
        now = self.CurrentDateTime
        seconds_since_midnight = now.hour * 3600 + now.minute * 60 + now.second # secで表示
        rtnList.append(seconds_since_midnight / 86400)
        indx += 1

        # 3: 外気乾球温度
        val = await self.envCom.get_drybulb_temperature()
        rtnList.append(val[1] / 40 if val[0] else self._prevRtn[indx])
        indx += 1

        # 4: 外気相対湿度
        val = await self.envCom.get_relative_humidity()
        rtnList.append(val[1] / 100 if val[0] else self._prevRtn[indx])
        indx += 1

        # 各ゾーンの温度
        i_unit_num = [5,4,5,4] # 各系統の室内機の台数
        for i in range(len(i_unit_num)):
            for j in range(i_unit_num[i]):
                val = await self.envCom.get_zone_drybulb_temperature(i+1,j+1)
                rtnList.append(val[1] / 40 if val[0] else self._prevRtn[indx])
                indx += 1 

        # 南側ゾーンの執務者数
        for i in range(1, 10):
            val = await self.occCom.get_zone_occupant_number(occ.Tenant.South, i)
            rtnList.append(val[1] / 10 if val[0] else self._prevRtn[indx])
            indx += 1
            val = await self.occCom.get_zone_occupant_number(occ.Tenant.North, i)
            rtnList.append(val[1] / 10 if val[0] else self._prevRtn[indx])
            indx += 1            

        self._prevRtn = rtnList # 出力を保存
        return np.array(rtnList)


    async def _calculate_reward(self, is_final_step):
        """
        報酬を計算する
        Args:
            is_final_step: エピソードの最終ステップか否か
        Returns:
            報酬
        """
        # 不満足者数 [人] (最大で100人程度, 基準乱数で86人)
        while not (result:= await self.occCom.get_occupant_number(self.occCom.Tenant.South))[0]:pass
        dis_occ = result[1]
        while not (result:= await self.occCom.get_occupant_number(self.occCom.Tenant.North))[0]:pass
        dis_occ += result[1]
        while not (result:= await self.envCom.get_instantaneous_dissatisfaction_rate())[0]:pass
        dis_occ *= result[1]

        # エネルギー消費量 [kW] (最大で35kW程度, 平均で8kW程度)
        while not (result:= await self.vrfCom.get_outdoor_unit_electricity(1))[0]:pass
        vrf_elec = result[1]
        while not (result:= await self.vrfCom.get_outdoor_unit_electricity(2))[0]:pass
        vrf_elec += result[1]
        while not (result:= await self.vrfCom.get_outdoor_unit_electricity(3))[0]:pass
        vrf_elec += result[1]
        while not (result:= await self.vrfCom.get_outdoor_unit_electricity(4))[0]:pass
        vrf_elec += result[1]
        
        # 不満足者数とエネルギー消費を重み付け平均
        rwd = -(2.0 * dis_occ / 86.0 + 1.0 * vrf_elec / 35.0) # ほぼ0-1の範囲にした上で重み付け平均
        if self.prev_rwd != 0:
            rwd = rwd + (rwd - self.prev_rwd) # Shaping: 前回の報酬との差分を加算

        self.ttl_rwd = self.ttl_rwd + rwd

        if is_final_step:
            with open('eci.csv', 'a', newline='') as f:
                while not (result:= await self.envCom.get_averaged_dissatisfaction_rate())[0]:pass
                self.disrate = result[1]
                while not (result:= await self.envCom.get_total_energy_consumption())[0]:pass
                self.energy = 0.001 * result[1]
                writer = csv.writer(f)
                is_summer = (self._final_dtime.month == 7)
                eci = self._calc_eci(self.energy, self.disrate, is_summer)
                writer.writerow([self.energy, self.disrate, self.ttl_rwd, eci])
        return rwd


    @staticmethod
    def _calc_eci(energy, disrate, is_summer):
        if is_summer:
            lmd_e = 0.8678
            myu_e = 3.9653
            sig_e = 1.5129
            lmd_d = -0.4356
            myu_d = -2.4076
            sig_d = 0.3343                
        else :
            lmd_e = 0.3899
            myu_e = 0.8598
            sig_e = 0.8777
            lmd_d = -1.3428
            myu_d = -5.4246
            sig_d = 1.0083                
        
        pe = 1 - norm.cdf(((energy ** lmd_e - 1) / lmd_e - myu_e) / sig_e)
        pd = 1 - norm.cdf(((disrate ** lmd_d - 1) / lmd_d - myu_d) / sig_d)
        return np.sqrt(pe * pd)

    # endregion
