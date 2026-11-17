"""Generador offline de referencias JSON (schema v1) para la app de rehabilitación.

La lógica pura (ángulos, ciclo, schema) vive en módulos framework-free; el video
y el modelo de pose entran sólo por :mod:`~src.references.backend` y
:mod:`~src.references.extract`. Entrada ejecutable: ``python -m src.references``.
"""
