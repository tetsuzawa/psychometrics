# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# ##################################################
# Best PEST法を用いた移動音の方向弁別の閾値推定
#
# 作成者:瀧澤哲
# 作成年:2020
# ##################################################

import numpy as np
import matplotlib.pyplot as plt


class BestPEST:
    """Best PEST法による実験のためのクラス

    Best PEST法 (or ML法, 最尤法) は心理物理測定法のうちの適応法の一つである.
    各施行ごとに最尤推定を行い, 閾値を推定する.


    参考:
        津村尚志 "最近の聴覚心理実験における新しい測定法"（1984）
        津村尚志 "聴覚実験における計量心理学の応用"（1986）
        津村尚志 "聴覚心理測定法の最近の動向"（1991）
        寺岡章人, 津村尚志 "訓練を受けた聴取者と受けていない聴取者を用いた三つの心理物理測定法の実験効率の比較"（1998）

    :ivar M: 刺激レベルの目標値(=Xt). 閾値. これを求める.
    :ivar S: 心理測定関数PFの広がりを表すパラメータ. 閾値. これを求める.
    :ivar T: 総試行回数.
    :ivar C: 総正答数.
    :ivar _XTCs: 刺激レベルと回答数・正答数の記録. 最尤推定に使用する.
    :ivar _T_end: 試行終了回数.
    :ivar _eta: 学習率.
    :ivar _eps: 最急降下法によってパラメータを推定するときのループ終了閾値.
    """

    def __init__(self, init_M=30.0, init_S=1.0, a=0.5, b=0.5, T_end=50, eta=0.05, eps=1e-6):
        """初期化関数

        :param init_M: 推定閾値（パラメータ）の初期値.
        :param init_S: 推定パラメータの初期値. Sの値は真値から1/4~4倍異なっていても, Mの推定値に影響しない.
        :param a: 心理測定関数の係数. 強制選択法（2IFC）なら0.5, Yes/No法なら1.
        :param b: 心理測定関数のバイアス. 強制選択法（2IFC）なら0.5, Yes/No法なら0.
        :param T_end: 試行終了回数.
        :param eta: 学習率.
        :param eps: 最急降下法によってパラメータを推定するときのループ終了閾値.
        """
        self.M = init_M
        self.S = init_S
        self._a = a
        self._b = b
        self.T = 0
        self.C = 0
        self._XTCs = np.empty([0, 3], float)
        self._T_end = T_end
        self._eta = eta
        self._eps = eps

    def update(self, is_correct: bool, X: float) -> float:
        """最尤推定と最急降下法によって閾値を推定し, 更新後の刺激レベルを返す.

        :param is_correct: 刺激レベル. 被験者の回答が正答か否か.
        :param X: 刺激レベル.
        :return: 更新後の刺激レベル.
        """
        self.T += 1
        if is_correct:
            self.C += 1
        self._XTCs = np.vstack([self._XTCs, [X, self.T, self.C]])

        M = self.M
        S = self.S
        params = np.array([M, S])
        for i in range(3000):
            # 勾配
            gradient = self._find_gradient()
            # パラメータの更新
            params = params - self._eta * gradient
            self.M = params[0]
            self.S = params[1]

            if (gradient ** 2).sum() < self._eps ** 2:
                break

        updated_X = self.M
        return updated_X

    def _find_gradient(self) -> (float, float):
        """対数尤度関数の勾配を求める

        :param stimulation_level_direction: 刺激レベルの変化方向.
        """
        # Sを最適化しない場合
        dls = np.array([[BestPEST._L_M(X, T, C, self.M, self.S, self._a, self._b), 0] for X, T, C in self._XTCs])
        # Sも最適化する場合（微分項が1/x^2のため計算時間が大幅に増えたり、勾配が乱れて推定に時間がかかったりする）
        # dls = np.array([[self._L_M(X, T, C, self.M, self.S, self._a, self._b),
        #                  self._L_S(X, T, C, self.M, self.S, self._a, self._b)] for X, T, C in self._XTCs])

        # 勾配
        gradient = np.sum(dls, axis=0)
        # 尤度関数（対数尤度関数）は上に凸なので、-1を掛ける
        gradient = -1 * gradient
        return gradient

    def has_ended(self) -> bool:
        """終了判定. 規定の試行回数で終了
        """
        return self.T == self._T_end

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
        z = BestPEST._Z(X, M, S)
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
        pf = BestPEST.PF(X, M, S, a, b)
        z_m = BestPEST._Z_M(S)
        return a * z_m * pf * (1 - pf)

    @staticmethod
    def _PF_S(X: float, M: float, S: float, a: float, b: float) -> float:
        """PFのSによる偏微分
        """
        pf = BestPEST.PF(X, M, S, a, b)
        z_s = BestPEST._Z_S(X, M, S)
        return a * z_s * pf * (1 - pf)

    @staticmethod
    def L(X: float, T: int, C: int, M: float, S: float, a: float, b: float) -> float:
        """対数尤度関数
        """
        pf = BestPEST.PF(X, M, S, a, b)
        t1 = C * np.log(pf)
        t2 = (T - C) * np.log(1 - pf)
        return t1 + t2

    @staticmethod
    def _L_M(X: float, T: int, C: int, M: float, S: float, a: float, b: float) -> float:
        """LのMによる偏微分
        """
        pf = BestPEST.PF(X, M, S, a, b)
        return (C - T * pf) / (pf * (1 - pf)) * BestPEST._PF_M(X, M, S, a, b)

    @staticmethod
    def _L_S(X: float, T: int, C: int, M: float, S: float, a: float, b: float) -> float:
        """LのSによる偏微分
        """
        pf = BestPEST.PF(X, M, S, a, b)
        return (C - T * pf) / (pf * (1 - pf)) * BestPEST._PF_S(X, M, S, a, b)

    # -------------------------- Example -------------------------- #
    @staticmethod
    def example():
        # 刺激レベルの対象範囲
        mock_x = list(range(1, 51))

        # 最大・最小刺激レベル
        max_X = 50
        min_X = 1
        # 刺激レベルの変化を記録するためのリスト
        X_list = []

        # 心理測定関数のパラメータの真値（強制選択法, 2IFC）
        true_M = 20.0
        true_S = 1.5
        true_a = 1 / 2  # 傾き1/2
        true_b = 1 / 2  # バイアス1/2
        true_Pt = 0.75
        # 試行終了回数
        T_end = 50

        # BestPEST法のインスタンス生成
        best_pest = BestPEST(T_end=T_end)

        # 試行回数T
        T = 0
        # 実験開始
        print("実験開始")
        while True:
            T += 1
            if T == 1:
                X = max_X
            elif T == 2:
                X = min_X

            if X < min_X:
                X = min_X
            elif X > max_X:
                X = max_X

            X_list.append(X)
            print(f"\rT:{T}, X:{X}")

            # 被験者の回答の正誤
            is_correct = np.random.rand() < BestPEST.PF(X, true_M, true_S, true_a, true_b)

            # 刺激レベルの更新
            X = int(best_pest.update(is_correct, X))

            # 実験終了判定
            if best_pest.has_ended():
                print("実験終了")
                break

        # 推定結果の出力
        true_Xt = BestPEST.PF_inv(true_Pt, true_M, true_S, true_a, true_b)
        print(f"刺激レベルの閾値. 真値: {true_Xt}, 推定値: {X}")

        # 心理測定関数の閾値をプロット
        y = [BestPEST.PF(x_tmp, true_M, true_S, true_a, true_b) for x_tmp in mock_x]
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
    BestPEST.example()
