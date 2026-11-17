"""Auditoría humana de pseudo-labels vía CVAT self-hosted.

Flujo completo (ver ``docs/cvat_setup.md``):

1. :mod:`src.audit.select_audit_subset` — selecciona qué frames auditar.
2. :mod:`src.audit.pack_for_cvat` — empaqueta imágenes + pre-anotaciones.
3. :mod:`src.audit.verify_roundtrip` — gate: la visibilidad debe sobrevivir
   el viaje CVAT ida y vuelta antes de corregir nada a mano.
4. :mod:`src.audit.reconcile_corrections` — compara correcciones humanas
   contra los pseudo-labels y produce el gold set + la métrica de tesis.

Todos los módulos son framework-free (stdlib + cv2), así que corren en
cualquier venv del repo.
"""
