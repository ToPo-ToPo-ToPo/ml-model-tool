"""Keys shared by the data sources (sample count, split, seed, the SIMP problem)."""
from __future__ import annotations

from ..keys import Choice, Int, Key, Num, Pair, Subset

SUPPORTS = ("cantilever", "both-ends", "pin-roller")


def n_key(default: int, help: tuple[str, str]) -> Key:
    return Key("n", default, Int(10), help)


def split_keys(n_test: int, n_val: int, what: tuple[str, str] = ("件", "samples")) -> list[Key]:
    return [Key("n_test", n_test, Int(0), (f"テストの{what[0]}数（学習に使わない。評価はこれで行う）",
                                           f"test {what[1]} (never trained on; evaluate uses them)")),
            Key("n_val", n_val, Int(0), (f"検証の{what[0]}数（毎エポックの検証の損失。最良のエポックの選択に使う）",
                                         f"validation {what[1]} (validation loss every epoch; picks the best epoch)")),
            Key("seed", 0, Int(0), ("乱数の種（同じ種なら同じデータ・同じテスト件）",
                                    "random seed (the same seed gives the same data and test cases)"))]


def simp_keys() -> list[Key]:
    """The SIMP problem on a rectangle (simp-topopt, simp-fea)."""
    return [
        Key("nelx", 64, Int(8, 512), ("横の要素数", "elements along x")),
        Key("nely", 32, Int(8, 512), ("縦の要素数", "elements along y")),
        Key("volfrac", [0.3, 0.5], Pair(0.05, 0.95), ("体積率の範囲 [最小, 最大]（1 件ごとにこの範囲から引く）",
                                                    "volume fraction range [min, max] (drawn per sample)")),
        Key("supports", list(SUPPORTS), Subset(SUPPORTS),
            ("拘束の種類: cantilever（左端を固定）・both-ends（左右端）・pin-roller（左下 3 節点を x・y、右下 3 節点を y だけ固定）",
             "support types: cantilever (left edge fixed), both-ends (left and right edges), pin-roller "
             "(3 bottom-left nodes fixed in x and y, 3 bottom-right nodes in y)"),
            ("3 種すべて", "all three")),
        Key("load_nodes", [2, 3], Pair(2, 8, integer=True),
            ("線荷重の節点数の範囲（境界上。合計の大きさ 1、向きはランダム。点荷重は 2 次元弾性で特異なので使わない）",
             "nodes of the line load (on the boundary, total magnitude 1, random direction; no point loads)")),
        Key("rmin", 1.5, Num(0, 10, lo_open=True), ("SIMP の密度フィルタの半径（要素数）", "SIMP density filter radius (elements)")),
        Key("maxiter", 200, Int(1, 10000), ("SIMP の最大反復回数", "SIMP iteration limit")),
        Key("workers", 1, Int(1, 256), ("並列のプロセス数", "parallel processes")),
    ]


def target_key(options: tuple, default: str, help: tuple[str, str]) -> Key:
    return Key("target", default, Choice(options), help)
