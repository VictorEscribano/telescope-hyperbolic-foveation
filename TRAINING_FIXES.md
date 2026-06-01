# Telescope — correcciones para entrenar acorde al paper

Revisión de consistencia con el paper (arXiv:2604.06332) y correcciones para que
el entrenamiento funcione correctamente. Estado vivo: marca lo hecho / pendiente.

## Contexto del problema original
El entrenamiento mostraba `matched=0` en todos los steps y `loss` plano (~1.9):
el modelo no veía ninguna caja GT. Causa raíz: la carga de anotaciones de
Argoverse2 fallaba en silencio (`except Exception` se tragaba el error).

---

## ✅ COMPLETADO

### 1. Carga de anotaciones (bug `matched=0`) — `telescope/data.py`
Tres bugs contra la API de `av2` 0.3.6, todos silenciados por `except Exception`:
- `get_camera_params` no existe → es `get_log_pinhole_camera(log_id, cam_name)`.
- `project_ego_to_img` devuelve **tupla `(uv, points_cam, is_valid)`**, no un array
  `(N,3)`; el check "delante de cámara" ahora usa `points_cam[:, 2] > 0`.
- `get_labels_at_lidar_timestamp` devuelve un `CuboidList`; hay que iterar
  `cuboid_list.cuboids`.
- Extra: `dst_SE3_object.translation` → `xyz_center_m`; el `except` ahora avisa
  con `warnings.warn` en vez de descartar en silencio.

### 2. Calentamiento de LR desperdiciaba la época 0 — `train.py`
El `LambdaLR` por época evaluaba la rampa en el índice 0 → factor `1e-8` (LR≈0
toda la primera época). Sustituido por **rampa lineal por iteración** sobre la
primera época (`set_lr(global_step)`), luego constante. Se eliminó el
`scheduler.step()` por época.

### 3. Ablación `--no_foveation` rota — `telescope/pipeline.py`, `train.py`
`forward()` nunca leía `_no_foveation`: la imagen se deformaba con la `R` del
estimador aleatorio congelado mientras la pérdida decodificaba con `R=0.001`
(warp/decode inconsistentes). Ahora `forward()` fuerza `R≈1e-3` → Φ=identidad
consistente en warp, embedding y decode. Verificado: `|c_eu − c_ri| = 0`.

### 4. Backbone real SAM 3.1 — `telescope/backbone_sam3.py` (NUEVO), `train.py`
`_load_sam3_backbone` era un `TODO` no-op; `--backbone_ckpt` se ignoraba y se
entrenaba con `SAM3EncoderStub` (convs 1×1 aleatorias congeladas).
- Nuevo `SAM3Backbone`: ViT (depth 32) + SimpleFPN neck reales, carga **solo** los
  pesos `detector.backbone.vision_backbone.*` del checkpoint multiplex (438 pesos,
  ignora 36 claves del neck interactivo SAM2).
- Neck con `scale_factors=(4.0,2.0,1.0)` → 3 niveles de 256 canales a 288²/144²/72²
  (img 1008); devueltos coarse→fine para casar con el contrato del stub.
- Normalización SAM3 `(x·2−1)` y resize a 1008 dentro del wrapper.
- Corre **siempre en eval** (override de `.train()`) para desactivar activation
  checkpointing y drop-path del ViT, y en **bf16 nativo** (el ViT da NaN en fp16),
  bajo `no_grad` cuando está congelado → no guarda activaciones del ViT.
- Verificado en GPU: carga limpia, **VRAM 9.1 GB** con batch 2 @ 1024 (GPU de 12 GB),
  loss baja, gradientes finitos a fov/emb/detr/box_head, backbone congelado.

### 5. Geometría inestable en fp16 (bug crítico latente) — `telescope/box.py`, `telescope/warp.py`
`EPS=1e-8` es subnormal en fp16 (**= 0.0**), así que `clamp(min=EPS)` no protegía y
`1/r`, `1/‖col‖` → inf → **grads NaN**. Estaba oculto porque con `matched=0` nunca
se ejercitaba el decode de cajas. Ahora `riemannian_to_euclidean_box`,
`euclidean_to_riemannian_box` y el grid del warp corren en **fp32 con autocast
desactivado**. También se bajó `init_scale` del GradScaler a `2**13`.

### 6. Denoising DINO-style — `telescope/pipeline.py`, `telescope/matcher.py`, `train.py`
`denoise_boxes` se importaba pero nunca se usaba. Implementado:
- `RealDeformableDetr`/`DeformableDetrStub` refactorizados en `encode`/`decode`
  para **reusar el encoder** (lo caro) entre el pase de matching y el de denoising.
- `TelescopeModel._denoising_pass`: genera queries de GT con ruido (label embedding
  + foveación, ref-points = centros ruidosos) y corre un **pase separado** del
  decoder (el decoder de HF no tiene máscara de self-attention → la concatenación
  filtraría las GT a las matching queries).
- `compute_denoising_loss` (matcher.py): pérdida L1+gIoU+cls directa 1-a-1, sin
  Hungarian.
- Flags en train.py: `--denoising` (ON por defecto), `--no_denoising`,
  `--dn_noise_scale 0.4`, `--dn_weight 1.0`.
- Verificado end-to-end: dn_loss baja, grad llega a `dn_label_emb`, path sin
  denoising sigue devolviendo la 5-tupla (retrocompatible).

---

## ⏳ PENDIENTE

### #3 — mAP por distancia (`telescope/eval.py`, `telescope/data.py`)
La tabla estrella del paper (0–50/50–150/150–250/250 m+) NO está implementada;
`summarize()` solo da splits por tamaño de COCO. `data.py` calcula la distancia
solo para filtrar (`max_dist`) y la descarta. Falta: propagar la distancia por
caja al `target`, y computar mAP por bins en `eval.py`.

### #4 — Gradiente de Φ⁻¹ a `o, R` (`telescope/geometry.py`)
`HyperbolicInverseNR.backward` devuelve `None` para `o, R`: la foveación no
aprende por el camino del warp/resampling (sí por embedding + Jacobiano del
decode). Extender el backward con el teorema de la función implícita para
propagar también a `o` y `R`. (Nota: con el backbone bajo `no_grad`, este camino
no aporta a través de las features; sí aportaría vía el decode si se cierra.)

### #5 — Menores (`telescope/eval.py`, `telescope/pipeline.py`)
- Cajas en `[-1,1]`: los thresholds small/medium/large de COCO (32²/96² px) no
  tienen sentido a esa escala (todo cae en "small").
- La estimación de parámetros usa 64×64 (`pipeline.py` Stage 1a); el paper/docstring
  dicen 256/512.

---

## Notas de verificación / entorno
- GPU del entorno: **12 GB** (el README dice 14 GB). Backbone real cabe con batch 2.
- `flash_attn` NO instalado; SAM3 corre con SDPA (`use_fa3=False`). Se instaló `einops`.
- El backbone real requiere `query_dim=256` (d_model del neck SAM3).
