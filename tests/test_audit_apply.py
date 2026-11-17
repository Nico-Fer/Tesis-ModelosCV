"""Dataset corregido y folds (:mod:`src.audit.apply_corrections`).

Mini-pool sintético de 5 frames / 4 sujetos / 2 folds que cubre los tres
efectos del gold sobre la membresía de los folds: **sumar** un frame (sin
detección en el pseudo, anotado de cero), **quitar** un frame (el anotador
borró todo) y **modificar** un label (membresía intacta). Todo lo demás tiene
que quedar idéntico a ``uco_pseudo``. Framework-free: corre en cualquier venv.

Ejecutar::

    python tests/test_audit_apply.py        # sin pytest
    pytest tests/                            # si pytest está instalado
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from src.audit.apply_corrections import apply
from src.audit.common import AuditConfig, build_config

_L_PSEUDO = "0 0.5 0.5 0.2 0.4" + " 0.5 0.5 2" * 17
_L_HUMANO = "0 0.5 0.5 0.2 0.4" + " 0.6 0.4 2" * 17
# Mismos keypoints que el pseudo; solo cambia el bbox (hull + 5 % del gold).
_L_CONFIRMADO = "0 0.5 0.5 0.1 0.1" + " 0.5 0.5 2" * 17

# stem → (label pseudo, label gold o None si no se auditó, n_kpts_corregidos)
_A = "0_01_cam0_000000"  # sujeto 0: sano, sin auditar
_B = "0_01_cam0_000030"  # sujeto 0: sin detección → anotado de cero (ENTRA)
_C = "1_01_cam0_000000"  # sujeto 1: el anotador borró todo (SALE)
_D = "2_01_cam0_000000"  # sujeto 2: label corregido (se queda, label humano)
_E = "3_01_cam0_000000"  # sujeto 3: sano, sin auditar
_F = "3_01_cam0_000030"  # sujeto 3: auditado SIN correcciones (conserva el pseudo)
_FRAMES = {
    _A: (_L_PSEUDO, None, 0),
    _B: ("", _L_HUMANO, 0),
    _C: (_L_PSEUDO, "", 0),
    _D: (_L_PSEUDO, _L_HUMANO, 17),
    _E: (_L_PSEUDO, None, 0),
    _F: (_L_PSEUDO, _L_CONFIRMADO, 0),
}
_MOTIVOS = {_B: "sin_deteccion", _C: "score_bajo", _D: "discrepancia_gt", _F: "control"}


def _escenario(tmp: Path, **aplicar: object) -> AuditConfig:
    pseudo = tmp / "uco_pseudo"
    img_dir, lbl_dir, folds = pseudo / "images" / "all", pseudo / "labels" / "all", pseudo / "folds"
    gold = tmp / "uco_gold"
    for d in (img_dir, lbl_dir, folds, gold / "labels" / "all"):
        d.mkdir(parents=True)
    for stem, (pseudo_lbl, gold_lbl, _) in _FRAMES.items():
        (img_dir / f"{stem}.jpg").write_bytes(b"jpg")
        (lbl_dir / f"{stem}.txt").write_bytes((pseudo_lbl + ("\n" if pseudo_lbl else "")).encode())
        if gold_lbl is not None:
            (gold / "labels" / "all" / f"{stem}.txt").write_bytes(
                (gold_lbl + ("\n" if gold_lbl else "")).encode()
            )

    # Reparto heredado: sujetos 0,1 → fold 0; 2,3 → fold 1. skip_empty: _B no está.
    def ruta(s: str) -> str:
        return str((img_dir / f"{s}.jpg").resolve())

    manifests = {
        "fold0_val": [_A, _C],
        "fold0_train": [_D, _E, _F],
        "fold1_val": [_D, _E, _F],
        "fold1_train": [_A, _C],
    }
    for nombre, stems in manifests.items():
        (folds / f"{nombre}.txt").write_text("".join(f"{ruta(s)}\n" for s in stems), "utf-8")

    with (gold / "gold_manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["stem", "motivo", "n_personas_gold", "n_kpts_corregidos"]
        )
        writer.writeheader()
        for stem, motivo in _MOTIVOS.items():
            gold_lbl, n_corr = _FRAMES[stem][1], _FRAMES[stem][2]
            writer.writerow(
                {
                    "stem": stem,
                    "motivo": motivo,
                    "n_personas_gold": str(1 if gold_lbl else 0),
                    "n_kpts_corregidos": str(n_corr),
                }
            )

    return build_config(
        {
            "pseudo": {
                "images_dir": str(img_dir),
                "labels_dir": str(lbl_dir),
                "folds_dir": str(folds),
            },
            "estratos": {"postura": {"01": "sedente"}},
            "pack": {"link_mode": "copy"},
            "reconcile": {"gold_root": str(gold)},
            "aplicar": {
                "out_root": str(tmp / "uco_pseudo_corrected"),
                "out_yaml": str(tmp / "pose_dataset_pseudo_corrected.yaml"),
                "n_folds": 2,
                **aplicar,
            },
        }
    )


def _stems(path: Path) -> list[str]:
    return [Path(line).stem for line in path.read_text(encoding="utf-8").split() if line]


def test_apply_suma_quita_y_corrige_sin_tocar_el_resto() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg = _escenario(tmp)
        summary = apply(cfg)
        root = cfg.corrected_root.resolve()
        folds = root / "folds"

        assert summary["n_pool"] == 6 and summary["n_gold"] == 4
        assert summary["n_gold_reemplazados"] == 3  # _B, _C, _D
        assert summary["n_gold_confirmados"] == 1  # _F
        assert _stems(folds / "fold0_val.txt") == [_A, _B]  # _B entra, _C sale
        assert _stems(folds / "fold0_train.txt") == [_D, _E, _F]
        assert _stems(folds / "fold1_val.txt") == [_D, _E, _F]
        assert _stems(folds / "fold1_train.txt") == [_A, _B]
        assert _stems(folds / "fold0_val_gold.txt") == [_B]
        assert _stems(folds / "fold1_val_gold.txt") == [_D, _F]

        # Labels: el gold pisa solo los corregidos; el resto (incluido el
        # auditado sin correcciones, _F) es byte a byte el pseudo.
        lbl = root / "labels" / "all"
        assert lbl.joinpath(f"{_D}.txt").read_text("utf-8").strip() == _L_HUMANO
        assert lbl.joinpath(f"{_C}.txt").read_text("utf-8") == ""
        pseudo_lbl = cfg.labels_dir
        for stem in (_A, _E, _F):
            assert (
                lbl.joinpath(f"{stem}.txt").read_bytes()
                == pseudo_lbl.joinpath(f"{stem}.txt").read_bytes()
            ), stem
        assert len(list((root / "images" / "all").glob("*.jpg"))) == 6

        with (root / "provenance.csv").open(encoding="utf-8", newline="") as fh:
            prov = {r["stem"]: (r["fuente"], r["label_de"]) for r in csv.DictReader(fh)}
        assert prov == {
            _A: ("pseudo", "pseudo"),
            _B: ("humano", "gold"),
            _C: ("humano", "gold"),
            _D: ("humano", "gold"),
            _E: ("pseudo", "pseudo"),
            _F: ("humano", "pseudo"),
        }

        for nombre in ("fold0", "fold1", "gold_fold0", "gold_fold1"):
            assert (tmp / f"pose_dataset_pseudo_corrected_{nombre}.yaml").exists(), nombre


def test_apply_solo_eval_excluye_gold_del_train() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _escenario(Path(tmpdir), usar_en_train=False)
        apply(cfg)
        folds = cfg.corrected_root.resolve() / "folds"
        assert _stems(folds / "fold0_train.txt") == [_E]  # _D es gold
        assert _stems(folds / "fold1_train.txt") == [_A]  # _B es gold
        assert _stems(folds / "fold0_val.txt") == [_A, _B], "el val conserva el gold"


def test_apply_membresia_ajena_al_gold_aborta() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _escenario(Path(tmpdir))
        train0 = cfg.folds_dir / "fold0_train.txt"
        train0.write_text(train0.read_text("utf-8").splitlines()[0] + "\n", "utf-8")  # quita _E
        try:
            apply(cfg)
        except ValueError as exc:
            assert "membresía" in str(exc)
            return
    raise AssertionError("Un frame no-gold que cambia de fold debe abortar.")


def test_apply_rechaza_destino_que_pisa_el_pseudo() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg = _escenario(tmp, out_root=str(tmp / "uco_pseudo"))
        try:
            apply(cfg)
        except ValueError:
            assert (tmp / "uco_pseudo" / "labels" / "all" / f"{_A}.txt").exists()
            return
    raise AssertionError("out_root = pool pseudo debe abortar antes de borrar nada.")


def test_apply_mix_coco() -> None:
    import yaml

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        coco = tmp / "coco_val2017_pose.txt"
        coco.write_text("x.jpg\n", "utf-8")
        cfg = _escenario(tmp, mix_coco={"manifest": str(coco), "fold": 0})
        apply(cfg)
        spec = yaml.safe_load(
            (tmp / "pose_dataset_pseudo_corrected_coco_fold0.yaml").read_text("utf-8")
        )
        root = cfg.corrected_root.resolve()
        assert spec["train"] == [str(root / "folds" / "fold0_train.txt"), str(coco.resolve())]
        assert spec["val"] == str(root / "folds" / "fold0_val.txt")
        assert spec["kpt_shape"] == [17, 3]


# --------------------------------------------------------------------------- #
# Runner sin pytest
# --------------------------------------------------------------------------- #


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests OK")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
