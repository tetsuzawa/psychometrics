# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# ##################################################
# PEST法を用いた移動音の方向弁別の閾値推定
#
# 作成者:瀧澤哲
# 作成年:2020
# ##################################################

import numpy as np
import matplotlib.pyplot as plt


class Hybrid:
    """PEST法による実験のためのクラス

    PEST法 (Parameter Estimation by Sequential Testing) は心理物理測定法のうちの適応法の一つである.
    (a)刺激レベルを変える時期と, (b)刺激レベルの変化規則によって刺激レベルを逐次的に変化させて閾値を推定する.

    参考:
        津村尚志 "最近の聴覚心理実験における新しい測定法"（1984）
        津村尚志 "聴覚実験における計量心理学の応用"（1986）
        津村尚志 "聴覚心理測定法の最近の動向"（1991）
        寺岡章人, 津村尚志 "訓練を受けた聴取者と受けていない聴取者を用いた三つの心理物理測定法の実験効率の比較"（1998）

    :cvar _lower: 刺激レベルの変化方向（難化方向）.
    :cvar _upper: 刺激レベルの変化方向（易化方向）.

    :ivar Xt: 刺激レベルの目標値Xt. 閾値. これを求める.
    :ivar T: 総試行回数.
    :ivar C: 総正答数.
    :ivar Pt: 刺激レベルの目標値Ltに対する特定応答の出現率. 閾値. 強制選択法（2IFC）なら0.75, Yes/No法なら0.5.
    :ivar dx: 刺激変化幅.
    :ivar _min_dx: 最小刺激変化幅. 刺激幅dxが最小刺激幅min_dxとなったときに実験を終了.
    :ivar _max_dx: 最大刺激変化幅. 被験者の混乱を避けるため定義.
    :ivar _doubled_dx: 刺激変化幅が2倍されたものかの情報を保持.
    :ivar _switched_direction_by_doubled_dx: 刺激変化方向がが2倍された刺激変化幅によるものか情報を保持.
    :ivar _pre_stimulation_level_direction: # 前回の刺激レベルの変化方向の情報を保持.
    :ivar _repeated_num: 連続で同じ回答が繰り返された回数. 最小値:1.
    :ivar _consecutive_T: 一定の刺激レベルXで繰り返された回答数.
    :ivar _consecutive_C: 一定の刺激レベルXで繰り返された正答数.
    :ivar _W:  W  # deviation limit. 1.0 <= W <= 2.0.
    """
    _lower = False
    _upper = True

    def __init__(self, init_M=30, min_dx=1, Pt=0.75, W=1.5, a=0.5, b=0.5, T_end=50, eta=0.05, eps=1e-6):
        """初期化関数

        :param init_dx: 刺激レベルの初期値. 最小刺激レベルの2^4以上が望ましい.
        :param min_dx: 最小刺激レベル. 変化幅が最小刺激レベルとなったときに実験を終了する.
        :param Pt: 刺激レベルの目標値Ltに対する特定応答の出現率. 閾値. 強制選択法（2IFC）なら0.75, Yes/No法なら0.5.
        :param W: Deviation limit. 刺激レベル更新の判断に使用する. 通常1.0~2.0（推奨値1.0）.
        """

        self.Xt = 0.
        self.Pt = Pt
        self.T = 0
        self.C = 0
        self.dx = 4 * min_dx  # 刺激変化幅.
        self._min_dx = min_dx  # 最小刺激変化幅. 刺激幅dxが最小刺激幅min_dxとなったときに実験を終了する.
        self._max_dx = 8 * min_dx
        self._doubled_dx = False
        self._switched_direction_by_doubled_dx = False
        self._pre_stimulation_level_direction = self._lower  # lower:False or upper:True
        self._repeated_num = 1  # 連続で同じ回答が繰り返された回数. 最小値:1,
        self._consecutive_T = 0  # 一定の刺激レベルXで繰り返された回答数
        self._consecutive_C = 0  # 一定の刺激レベルXで繰り返された正答数
        self._W = W  # deviation limit. 1.0 <= W <= 2.0
        # ML
        self.M = init_M
        self._init_M = init_M
        self.S = min_dx  # init_S=min)dx
        self._init_S = min_dx
        self._a = a
        self._b = b
        self._XTCs = np.empty([0, 3], float)
        self._T_end = T_end
        self._eta = eta  # 学習率
        self._eps = eps  # 学習のストップ判定に用いる定数

    def update(self, is_correct: bool, X: float) -> float:
        """(a)刺激レベルを変える時期を判定し, 更新後の刺激レベルを返す.

        :param is_correct: 刺激レベル.被験者の回答が正答か否か.
        :param X: 刺激レベル.
        :return: 更新後の刺激レベル.
        """
        self.T += 1
        self._consecutive_T += 1
        if is_correct:
            self.C += 1
            self._consecutive_C += 1
        self._XTCs = np.vstack([self._XTCs, [X, self.T, self.C]])

        W = self._W
        I = self.Pt * self._consecutive_T - self._consecutive_C

        if np.abs(I) < W:
            self.Xt = X
            return self.Xt
        elif I <= -W:
            direction = self._lower
        elif I >= W:
            direction = self._upper
        else:
            raise ArithmeticError("Unknown arithmetic error occurred")

        self._consecutive_T = 0
        self._consecutive_C = 0
        self._find_dx(direction)
        self._pre_stimulation_level_direction = direction

        if self.dx > self._max_dx:
            self.dx = self._max_dx

        if direction == self._lower:
            self.Xt = X - self.dx
        else:
            self.Xt = X + self.dx
        return self.Xt

    def _find_dx(self, stimulation_level_direction: bool):
        """(b)刺激レベルの変化規則に乗っ取って, 刺激レベルの変化幅を更新する.

        津村尚志 "最近の聴覚心理実験における新しい測定法"（1984） の手順を参考にしている.

        :param stimulation_level_direction: 刺激レベルの変化方向.
        """
        # (1) 前回との刺激レベルの変化方向を比較
        if stimulation_level_direction != self._pre_stimulation_level_direction:
            if self._doubled_dx:
                self._switched_direction_by_doubled_dx = True
            self._repeated_num = 1
            self.dx = int(self.dx / 2)

        else:
            self._repeated_num += 1
            # (2)
            if self._repeated_num == 2:
                # self.dx = self.dx  # プログラム的に無駄なのでコメントアウト
                self._doubled_dx = False
            # (3)
            elif self._repeated_num >= 4:
                self.dx = 2 * self._repeated_num
                self._doubled_dx = True
            # (4)
            elif self._repeated_num == 3:
                if self._switched_direction_by_doubled_dx:
                    # self.dx = self.dx  # プログラム的に無駄なのでコメントアウト
                    self._doubled_dx = False
                else:
                    self.dx = 2 * self._repeated_num
                    self._doubled_dx = False

    def _calc_ML(self):
        self.M = self.Xt
        params = np.array([self.M, self.S])
        for i in range(300000):
            # 勾配
            gradient = self._find_gradient()
            # パラメータの更新
            params = params - self._eta * gradient
            self.M = params[0]
            self.S = params[1]

            if (gradient ** 2).sum() < self._eps ** 2:
                break

    def _find_gradient(self) -> (float, float):
        """対数尤度関数の勾配を求める

        :param stimulation_level_direction: 刺激レベルの変化方向.
        """
        # Sを最適化しない場合
        # dls = np.array([[Hybrid._L_M(X, T, C, self.M, self.S, self._a, self._b), 0] for X, T, C in self._XTCs])
        # Sも最適化する場合（微分項が1/x^2のため計算時間が大幅に増えたり、勾配が乱れて推定に時間がかかったりする）
        dls = np.array([[self._L_M(X, T, C, self.M, self.S, self._a, self._b),
                         self._L_S(X, T, C, self.M, self.S, self._a, self._b)] for X, T, C in self._XTCs])

        # 勾配
        gradient = np.sum(dls, axis=0)
        # 尤度関数（対数尤度関数）は上に凸なので、-1を掛ける
        gradient = -1 * gradient
        return gradient

    def has_ended(self) -> bool:
        """終了判定. 規定の試行回数で終了
        """
        if self.T == self._T_end:
            self._calc_ML()
            return True

        return False

    def validate_parameter(self):
        if self.S > 10 * self._init_S \
                or self.S < self._init_S / 10 \
                or abs(self.M - self._init_M) > 10 * self._init_S:
            print("初期値M0, S0の設定に誤りがある可能性があります。"
                  "この実験結果は採用するべきではありません")

    @staticmethod
    def _Z(X: float, M: float, S: float) -> float:
        return (X - M) / S

    @staticmethod
    def _Z_M(S: float) -> float:
        return - 1 / S

    @staticmethod
    def _Z_S(X: float, M: float, S: float) -> float:
        return - 1 * (X - M) / S ** 2

    @staticmethod
    def PF(X: float, M: float, S: float, a: float, b: float) -> float:
        """心理測定関数（ロジスティックス曲線と仮定）
        """
        sigmoid_range = 34.538776394910684
        z = Hybrid._Z(X, M, S)
        # オーバーフローを避ける
        if z <= -sigmoid_range:
            return 1e-15
        if z >= sigmoid_range:
            return 1.0 - 1e-15
        pf = a / (1 + np.exp(-z)) + b
        return pf

    @staticmethod
    def PF_inv(P: float, M: float, S: float, a: float, b: float) -> float:
        """心理測定関数の逆関数
        """
        pf_inv = M - S * np.log(a / (P - b) - 1)
        return pf_inv

    @staticmethod
    def _PF_M(X: float, M: float, S: float, a: float, b: float) -> float:
        """PFのMによる偏微分
        """
        pf = Hybrid.PF(X, M, S, a, b)
        z_m = Hybrid._Z_M(S)
        return a * z_m * pf * (1 - pf)

    @staticmethod
    def _PF_S(X: float, M: float, S: float, a: float, b: float) -> float:
        """PFのSによる偏微分
        """
        pf = Hybrid.PF(X, M, S, a, b)
        z_s = Hybrid._Z_S(X, M, S)
        return a * z_s * pf * (1 - pf)

    @staticmethod
    def L(X: float, T: int, C: int, M: float, S: float, a: float, b: float) -> float:
        """対数尤度関数
        """
        pf = Hybrid.PF(X, M, S, a, b)
        t1 = C * np.log(pf)
        t2 = (T - C) * np.log(1 - pf)
        return t1 + t2

    @staticmethod
    def _L_M(X: float, T: int, C: int, M: float, S: float, a: float, b: float) -> float:
        """LのMによる偏微分
        """
        pf = Hybrid.PF(X, M, S, a, b)
        return (C - T * pf) / (pf * (1 - pf)) * Hybrid._PF_M(X, M, S, a, b)

    @staticmethod
    def _L_S(X: float, T: int, C: int, M: float, S: float, a: float, b: float) -> float:
        """LのSによる偏微分
        """
        pf = Hybrid.PF(X, M, S, a, b)
        return (C - T * pf) / (pf * (1 - pf)) * Hybrid._PF_S(X, M, S, a, b)

    # -------------------------- Example -------------------------- #
    @staticmethod
    def example():
        """刺激レベルの閾値を推定する例

        心理測定関数PFの閾値Ptに対する刺激レベルの閾値Xtを推定する.
        """

        # 刺激レベルの対象範囲
        mock_x = list(range(1, 50))

        # 最大・最小刺激レベル
        min_X = 2
        init_M = 25
        # 刺激レベルの変化を記録するためのリスト
        X_list = []

        # 心理測定関数のパラメータの真値（強制選択法, 2IFC）
        true_M = 20
        true_S = 3
        true_a = 1 / 2  # 傾き1/2
        true_b = 1 / 2  # バイアス1/2
        true_Pt = 0.75
        T_end = 50

        # PEST法のインスタンス生成
        hybrid = Hybrid(init_M=init_M, min_dx=1, Pt=true_Pt, W=1.5, a=true_a, b=true_b, T_end=T_end)

        X = init_M + 4 * min_X
        # 試行回数T
        T = 0
        # 実験開始
        print("実験開始")
        while True:
            T += 1
            X_list.append(X)
            # 被験者の回答の正誤
            is_correct = np.random.rand() < Hybrid.PF(X, true_M, true_S, true_a, true_b)

            # 刺激レベルの更新
            X = hybrid.update(is_correct, X)

            # 実験終了判定
            if hybrid.has_ended():
                print("実験終了")
                break

        hybrid.validate_parameter()
        X = hybrid.M

        # 推定結果の出力
        true_Xt = Hybrid.PF_inv(true_Pt, true_M, true_S, true_a, true_b)
        print(f"{T}回目の回答で実験が終了しました.")
        print(f"刺激レベルの閾値. 真値: {true_Xt}, 推定値: {X}")

        # 心理測定関数の閾値をプロット
        y = [Hybrid.PF(x_tmp, true_M, true_S, true_a, true_b) for x_tmp in mock_x]
        plt.plot(mock_x, y, color="red", label=f"True PF")
        plt.hlines(y=true_Pt, xmin=0, xmax=true_Xt, linestyles="--")
        plt.vlines(x=true_Xt, ymin=1 / 2, ymax=true_Pt, linestyles="--", label="True threshold")
        plt.plot([X], [true_Pt], "o", color="orange", label="Estimated threshold")
        plt.xlabel("Stimulation level X")
        plt.ylabel("PF(X)")
        plt.title("Psychometric Function PF")
        plt.legend()
        plt.show()

        # 刺激レベルの軌跡
        plt.plot(list(range(1, T + 1)), X_list, "o-")
        plt.hlines(y=true_Xt, xmin=0, xmax=T + 1, linestyles="--")
        plt.xlabel("Numbers of trials T")
        plt.ylabel("Stimulation level X")
        plt.title("Path of stimulation level")
        plt.show()
    # -------------------------- Example -------------------------- #


if __name__ == '__main__':
    Hybrid.example()
