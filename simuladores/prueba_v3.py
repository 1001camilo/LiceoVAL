"""
Software Optico de Laboratorio v3
Cambios respecto a la version v2:
  - CORRECCIÓN: Las ecuaciones en la tercera pestaña ahora se renderizan
    correctamente sin superponerse, ajustando el espaciado y las cajas de texto.
  - NUEVO OBJETO: Se ha añadido un "Automóvil Sedán" (croquis de líneas negras
    de perfil) como opción de objeto asimétrico.
  - Ajustes menores de estilo para mejorar la legibilidad.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.widgets as widgets
from scipy.ndimage import gaussian_filter

# ══════════════════════════════════════════════════════════
#  ÁREA 1: ESTILO
# ══════════════════════════════════════════════════════════
ESTILO = {
    'color_fondo':      '#FAFAFA',
    'color_eje':        '#2C3E50',
    'font_family':      'sans-serif',
    'color_luz_fuente': '#FFF59D',
    'color_r1':         '#E74C3C',   # Rayo 1 - rojo
    'color_r2':         '#2980B9',   # Rayo 2 - azul
    'color_r3':         '#27AE60',   # Rayo 3 - verde
    'color_virtual':    '#F39C12',   # extensiones virtuales - naranja
    'color_lente':      '#85C1E9',
    'color_pantalla':   '#BDC3C7',
    'color_acetato':    '#7F8C8D',
    'color_objeto':     '#27AE60',
    'color_imagen':     '#C0392B',
    'f_ref':            10.0,        # focal de referencia para escalar el lente
    'alpha_lente':      0.50,
}

plt.rcParams.update({
    'figure.facecolor': ESTILO['color_fondo'],
    'axes.facecolor':   ESTILO['color_fondo'],
    'font.family':      ESTILO['font_family'],
    'axes.edgecolor':   ESTILO['color_eje'],
    'text.color':       ESTILO['color_eje'],
    # Habilitar mathtext para un mejor renderizado de ecuaciones sin LaTeX externo
    'mathtext.fontset': 'cm',
})

# ══════════════════════════════════════════════════════════
#  ÁREA 2: HELPERS DE DIBUJO PARA ACETATOS (vectorizados)
# ══════════════════════════════════════════════════════════
def _tline(img, x0, y0, x1, y1, t):
    """Traza una linea gruesa (radio t) de (x0,y0) a (x1,y1) - vectorizado."""
    s = img.shape[0]
    c, r = np.meshgrid(np.arange(s), np.arange(s))   # c=col=x, r=row=y
    dx, dy = x1 - x0, y1 - y0
    l2 = dx*dx + dy*dy
    if l2 < 1e-9: return img
    tp = np.clip(((c - x0)*dx + (r - y0)*dy) / l2, 0.0, 1.0)
    dist2 = (c - x0 - tp*dx)**2 + (r - y0 - tp*dy)**2
    img[dist2 < t*t] = 0.0
    return img

def _trifill(img, v1, v2, v3):
    """Rellena el triangulo definido por tres vertices (x,y)."""
    s = img.shape[0]
    c, r = np.meshgrid(np.arange(s), np.arange(s))
    def cr(ax, ay, bx, by):
        return (ax - bx)*(r - by) - (ay - by)*(c - bx)
    d1 = cr(v1[0], v1[1], v2[0], v2[1])
    d2 = cr(v2[0], v2[1], v3[0], v3[1])
    d3 = cr(v3[0], v3[1], v1[0], v1[1])
    dentro = ((d1>=0)&(d2>=0)&(d3>=0)) | ((d1<=0)&(d2<=0)&(d3<=0))
    img[dentro] = 0.0
    return img

def _circle(img, x0, y0, r, t):
    """Dibuja un circulo hueco centrado en (x0, y0) con radio r y grosor t."""
    s = img.shape[0]
    c, row = np.meshgrid(np.arange(s), np.arange(s))
    dist = np.sqrt((c - x0)**2 + (row - y0)**2)
    mask = (dist >= r - t) & (dist <= r + t)
    img[mask] = 0.0
    return img

# ══════════════════════════════════════════════════════════
#  ÁREA 3: GENERACION DE ACETATOS (objetos asimetricos)
# ══════════════════════════════════════════════════════════
def generar_letra_f(size=200):
    """Letra F — asimetrica en ambos ejes."""
    img = np.ones((size, size)); s = float(size); t = int(s/13)
    _tline(img, s*.23, s*.10, s*.23, s*.90, t)   # barra vertical
    _tline(img, s*.23, s*.10, s*.80, s*.10, t)   # barra superior (larga)
    _tline(img, s*.23, s*.50, s*.66, s*.50, t)   # barra media (corta)
    return img

def generar_flecha_diagonal(size=200):
    """Flecha apuntando arriba-derecha.
    Al invertirse (imagen real) apunta abajo-izquierda -> inversion evidente."""
    img = np.ones((size, size)); s = float(size); t = int(s/13)
    # Cuerpo de la flecha (diagonal de abajo-izq a centro)
    _tline(img, s*.10, s*.88, s*.55, s*.43, t)
    # Punta triangular
    tip = (int(s*.87), int(s*.12))
    bx, by = s*.55, s*.43
    dx = s*.87 - bx; dy = s*.12 - by
    lng = max(np.sqrt(dx**2 + dy**2), 1e-6)
    px, py = -dy/lng, dx/lng        # perpendicular unitario
    hw = s * .115
    v2 = (int(bx + hw*px), int(by + hw*py))
    v3 = (int(bx - hw*px), int(by - hw*py))
    _trifill(img, tip, v2, v3)
    return img

def generar_letra_j(size=200):
    """Letra J — gancho asimetrico, muestra inversion horizontal y vertical."""
    img = np.ones((size, size)); s = float(size); t = int(s/13)
    _tline(img, s*.30, s*.10, s*.72, s*.10, t)   # barra superior
    _tline(img, s*.60, s*.10, s*.60, s*.75, t)   # barra vertical (derecha)
    _tline(img, s*.60, s*.75, s*.44, s*.89, t)   # curva inferior parte 1
    _tline(img, s*.44, s*.89, s*.24, s*.83, t)   # curva inferior parte 2
    _tline(img, s*.24, s*.83, s*.17, s*.65, t)   # extremo del gancho
    return img

def generar_carro_sedan(size=200):
    """Automóvil Sedan (croquis de líneas negras de perfil)."""
    img = np.ones((size, size))
    s = float(size)
    t = int(s / 20)  # Grosor de línea
    t_rueda = int(s / 25)
    
    # Coordenadas relativas (0.0 a 1.0)
    y_suelo = 0.80
    y_chasis = 0.70
    y_capo = 0.55
    y_techo = 0.40
    x_ini = 0.10
    x_fin = 0.90
    x_rueda_del = 0.25
    x_rueda_tras = 0.75
    r_rueda = 0.10

    # --- Ruedas ---
    _circle(img, s*x_rueda_del, s*y_chasis, s*r_rueda, t_rueda)
    _circle(img, s*x_rueda_tras, s*y_chasis, s*r_rueda, t_rueda)
    
    # --- Chasis Inferior ---
    _tline(img, s*(x_ini + 0.05), s*y_chasis, s*(x_rueda_del - r_rueda + 0.02), s*y_chasis, t) # Frente bajo
    _tline(img, s*(x_rueda_del + r_rueda - 0.02), s*y_chasis, s*(x_rueda_tras - r_rueda + 0.02), s*y_chasis, t) # Entre ruedas
    _tline(img, s*(x_rueda_tras + r_rueda - 0.02), s*y_chasis, s*(x_fin - 0.05), s*y_chasis, t) # Trasera baja

    # --- Contorno Superior ---
    # Capó delantero
    _tline(img, s*x_ini, s*y_capo, s*0.30, s*y_capo, t)
    # Parabrisas delantero
    _tline(img, s*0.30, s*y_capo, s*0.40, s*y_techo, t)
    # Techo
    _tline(img, s*0.40, s*y_techo, s*0.70, s*y_techo, t)
    # Parabrisas trasero / Maletero
    _tline(img, s*0.70, s*y_techo, s*0.85, s*y_capo, t)
    # Tapa maletero
    _tline(img, s*0.85, s*y_capo, s*x_fin, s*y_capo, t)

    # --- Uniones Verticales (Frontal y Trasera) ---
    _tline(img, s*x_ini, s*y_capo, s*(x_ini+0.02), s*y_chasis, t) # Frente
    _tline(img, s*x_fin, s*y_capo, s*(x_fin-0.02), s*y_chasis, t) # Trasera

    # --- Ventanas ---
    t_win = int(t * 0.7)
    # Pilar B (central)
    _tline(img, s*0.55, s*y_techo, s*0.55, s*y_chasis, t_win)
    # Marco inferior ventanas
    _tline(img, s*0.32, s*y_capo, s*0.83, s*y_capo, t_win)

    return img

ACETATOS = {
    'Letra F':   generar_letra_f(),
    'Flecha':    generar_flecha_diagonal(),
    'Letra J':   generar_letra_j(),
    'Auto Sedan': generar_carro_sedan(), # Nuevo objeto añadido
}

# ══════════════════════════════════════════════════════════
#  ÁREA 4: FISICA Y RENDERIZADO
# ══════════════════════════════════════════════════════════
def calcular_fisica(so, f_mag, tipo_lente, x_pantalla, acetato_nombre):
    f_lente = f_mag if tipo_lente == 'Convexo' else -f_mag
    if abs(so - f_lente) < 0.01: so += 0.02

    try:
        si = 1.0 / (1.0/f_lente - 1.0/so)
    except ZeroDivisionError:
        si = np.inf

    M       = -si / so if si != np.inf else 1.0
    es_real = si > 0
    delta_x = abs(x_pantalla - si) if es_real else np.inf

    obj = ACETATOS[acetato_nombre]
    if es_real:
        sigma    = delta_x * 0.8
        img_proy = np.flipud(np.fliplr(obj.copy())) if M < 0 else obj.copy()
        if   sigma > 20:   img_proy = np.ones_like(img_proy) * 0.85
        elif sigma > 0.01: img_proy = gaussian_filter(img_proy, sigma=sigma)
    else:
        sigma    = x_pantalla * 0.3 + 1.0
        img_proy = gaussian_filter(obj, sigma=sigma) * 0.7 + 0.3

    return {'so': so, 'si': si, 'f': f_lente, 'M': M,
            'es_real': es_real, 'delta_x': delta_x}, img_proy

# ══════════════════════════════════════════════════════════
#  ÁREA 5: LAYOUT DE LA FIGURA
# ══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 9))
fig.canvas.manager.set_window_title("Software Optico de Laboratorio v3")

# ── Zonas graficas ────────────────────────────────────────
ax_bench  = fig.add_axes([0.05, 0.55, 0.90, 0.40])   # banco optico (arriba)
ax_object = fig.add_axes([0.05, 0.08, 0.165, 0.37])  # vista objeto (abajo-izq)
ax_screen = fig.add_axes([0.23, 0.08, 0.165, 0.37])  # vista pantalla (abajo-izq2)
ax_panel  = fig.add_axes([0.43, 0.08, 0.28,  0.37])  # panel informacion

# ── Tres pestanas sobre el panel ─────────────────────────
ax_tab1 = fig.add_axes([0.43,  0.462, 0.090, 0.038])
ax_tab2 = fig.add_axes([0.524, 0.462, 0.090, 0.038])
ax_tab3 = fig.add_axes([0.618, 0.462, 0.090, 0.038])

# ── Controles (derecha) ──────────────────────────────────
ax_radio_lente = fig.add_axes([0.755, 0.37, 0.095, 0.115])
# Ajustado el tamaño para acomodar la nueva opción
ax_radio_obj   = fig.add_axes([0.860, 0.32, 0.095, 0.165]) 
ax_so = fig.add_axes([0.775, 0.265, 0.170, 0.022])
ax_f  = fig.add_axes([0.775, 0.195, 0.170, 0.022])
ax_xs = fig.add_axes([0.775, 0.125, 0.170, 0.022])

# Etiquetas de sliders (texto fijo)
for yp, txt in [(0.292,'Distancia objeto  so (cm)'),
                (0.222,'Distancia focal   |f| (cm)'),
                (0.152,'Posicion pantalla xs (cm)')]:
    fig.text(0.775, yp, txt, fontsize=7.5, color=ESTILO['color_eje'])

# ══════════════════════════════════════════════════════════
#  ÁREA 6: WIDGETS
# ══════════════════════════════════════════════════════════
estado_app = {'tab': 'datos'}

COLOR_TAB_ON  = '#D6EAF8'
COLOR_TAB_OFF = '#E8E8E8'

btn_tab1 = widgets.Button(ax_tab1, 'Datos',      color=COLOR_TAB_ON,  hovercolor='#AED6F1')
btn_tab2 = widgets.Button(ax_tab2, 'Analisis',   color=COLOR_TAB_OFF, hovercolor='#AED6F1')
btn_tab3 = widgets.Button(ax_tab3, 'Ecuaciones', color=COLOR_TAB_OFF, hovercolor='#AED6F1')
for b in (btn_tab1, btn_tab2, btn_tab3):
    b.label.set_fontsize(8.5)

# Convexo / Concavo
radio_lente = widgets.RadioButtons(ax_radio_lente, ['Convexo', 'Concavo'])
# Lista de objetos actualizada
radio_obj   = widgets.RadioButtons(ax_radio_obj,   list(ACETATOS.keys()))
ax_radio_lente.set_title('Lente',   fontsize=8, pad=2)
ax_radio_obj.set_title('Objeto',    fontsize=8, pad=2)
for rb in (radio_lente, radio_obj):
    for lbl in rb.labels: lbl.set_fontsize(9)

sl_so = widgets.Slider(ax_so, 'so', 5.0,  60.0, valinit=25.0, valfmt='%.1f cm')
sl_f  = widgets.Slider(ax_f,  '|f|',5.0,  30.0, valinit=10.0, valfmt='%.1f cm')
sl_xs = widgets.Slider(ax_xs, 'xs', 0.0,  80.0, valinit=40.0, valfmt='%.1f cm')
for sl in (sl_so, sl_f, sl_xs):
    sl.label.set_fontsize(8); sl.valtext.set_fontsize(8)

# ══════════════════════════════════════════════════════════
#  ÁREA 7: FUNCIONES DE DIBUJO
# ══════════════════════════════════════════════════════════

# ── 7a. Vista frontal del objeto (acetato original) ───────
def dibujar_vista_objeto(nombre):
    ax_object.clear(); ax_object.axis('off')
    ax_object.imshow(ACETATOS[nombre], cmap='gray', vmin=0, vmax=1,
                     aspect='equal', interpolation='nearest')
    ax_object.add_patch(plt.Rectangle((0,0), 1, 1,
        transform=ax_object.transAxes,
        fill=False, edgecolor='#27AE60', lw=2.5))
    ax_object.set_title('OBJETO\n(Acetato original)',
                         fontsize=9, weight='bold', color='#27AE60', pad=4)
    # Flecha indicadora de orientacion
    ax_object.annotate('', xy=(0.85,0.85), xytext=(0.15,0.15),
        xycoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.2, alpha=0.5))
    ax_object.text(0.92, 0.92, 'orig', ha='right', va='top',
                   transform=ax_object.transAxes,
                   fontsize=7, color='#27AE60', alpha=0.7)

# ── 7b. Vista frontal de la pantalla ─────────────────────
def dibujar_vista_pantalla(img_proy, data):
    ax_screen.clear(); ax_screen.axis('off')
    ax_screen.imshow(img_proy, cmap='gray', vmin=0, vmax=1,
                     aspect='equal', interpolation='nearest')
    ax_screen.add_patch(plt.Rectangle((0,0), 1, 1,
        transform=ax_screen.transAxes,
        fill=False, edgecolor=ESTILO['color_pantalla'], lw=2.5))

    if data['es_real']:
        nitida = data['delta_x'] < 0.5
        color  = '#27AE60' if nitida else '#E74C3C'
        estado = 'NITIDA' if nitida else f'BORROSA  (Δ={data["delta_x"]:.1f} cm)'
        titulo = f'PANTALLA — {estado}'
        orientacion = 'Invertida' if data['M'] < 0 else 'Derecha'
        tamano      = 'Aumentada' if abs(data['M'])>1 else 'Reducida'
        ax_screen.set_title(titulo, fontsize=9, weight='bold',
                             color=color, pad=4)
        ax_screen.text(0.5, -0.04,
            f'{orientacion}  |  {tamano}  |  M = {data["M"]:.2f}',
            ha='center', va='top', transform=ax_screen.transAxes,
            fontsize=7.5, color=ESTILO['color_eje'])
    else:
        ax_screen.set_title('PANTALLA — Sombra difusa\n(imagen virtual, no proyectable)',
                             fontsize=8.5, color='#7F8C8D', pad=4)

# ── 7c. Panel de informacion (3 pestanas) ─────────────────
def dibujar_panel(data, xs):
    ax_panel.clear(); ax_panel.axis('off')
    ax_panel.add_patch(plt.Rectangle((0,0),1,1, transform=ax_panel.transAxes,
        facecolor='#FFFFFF', edgecolor='#BDC3C7', lw=1.5, zorder=0))

    tab = estado_app['tab']
    btn_tab1.ax.set_facecolor(COLOR_TAB_ON  if tab=='datos'      else COLOR_TAB_OFF)
    btn_tab2.ax.set_facecolor(COLOR_TAB_ON  if tab=='analisis'   else COLOR_TAB_OFF)
    btn_tab3.ax.set_facecolor(COLOR_TAB_ON  if tab=='ecuaciones' else COLOR_TAB_OFF)

    # ── Tab 1: Datos ────────────────────────────────────
    if tab == 'datos':
        ax_panel.text(0.5, 0.97, 'TABLA DE MEDICIONES OPTICAS',
                      ha='center', va='top', weight='bold', fontsize=10.5)
        ax_panel.axhline(0.92, color='#BDC3C7', lw=1, xmin=0.03, xmax=0.97)

        bloques = [
            ('Parametros del sistema', [
                ('Distancia objeto',  f"so = {data['so']:.2f} cm"),
                ('Distancia focal',   f"f  = {data['f']:.2f} cm"),
                ('Posicion pantalla', f"xs = {xs:.2f} cm"),
            ]),
            ('Resultados calculados', [
                ('Distancia imagen',  f"si = {data['si']:.2f} cm"),
                ('Magnificacion',     f"M  = {data['M']:.3f}"),
                ('Desfase enfoque',   f"Δx = {data['delta_x']:.2f} cm"),
            ]),
            ('Clasificacion', [
                ('Tipo imagen',  'Real (proyectable)' if data['es_real'] else 'Virtual'),
                ('Orientacion',  'Invertida' if data['M']<0 else 'Derecha'),
                ('Tamano',       'Aumentada' if abs(data['M'])>1 else 'Reducida'),
            ]),
        ]

        y = 0.88
        for titulo_bloque, filas in bloques:
            ax_panel.text(0.05, y, titulo_bloque, va='top', weight='bold',
                          fontsize=9.5, color='#2980B9')
            y -= 0.055
            for clave, valor in filas:
                ax_panel.text(0.08, y, clave + ':', va='top', fontsize=9,
                              family='monospace', color='#555555')
                ax_panel.text(0.95, y, valor, va='top', ha='right', fontsize=9.5,
                              family='monospace', weight='bold',
                              color=ESTILO['color_eje'])
                y -= 0.075
            y -= 0.03

    # ── Tab 2: Analisis fisico ───────────────────────────
    elif tab == 'analisis':
        ax_panel.text(0.5, 0.97, 'ANALISIS DEL FENOMENO',
                      ha='center', va='top', weight='bold', fontsize=10.5)
        ax_panel.axhline(0.92, color='#BDC3C7', lw=1, xmin=0.03, xmax=0.97)

        if data['es_real']:
            color_tit = '#27AE60'
            titulo_f  = 'Lente Convexo — Imagen Real'
            cuerpo = (
                "Los rayos que atraviesan las zonas\n"
                "transparentes del acetato convergen\n"
                "en un punto exacto a distancia si.\n\n"
                "Si la pantalla esta en si: imagen\n"
                "NITIDA (contraste tinta/luz).\n\n"
                "Si la pantalla esta lejos de si:\n"
                "aparecen circulos de confusion\n"
                "-> imagen BORROSA.\n\n"
                "La imagen real siempre esta\n"
                "INVERTIDA cuando so > f."
            )
        else:
            color_tit = '#E67E22'
            titulo_f  = 'Imagen Virtual — No proyectable'
            cuerpo = (
                "El lente Concavo (o so < f en\n"
                "Convexo) hace que los rayos\n"
                "divergan al salir del cristal.\n\n"
                "La imagen se forma del mismo\n"
                "lado que el objeto, no puede\n"
                "proyectarse en una pantalla.\n\n"
                "En la pantalla se ve una\n"
                "SOMBRA DIFUSA que aumenta\n"
                "con la distancia xs.\n\n"
                "Siempre: derecha y reducida."
            )

        ax_panel.text(0.5, 0.88, titulo_f, ha='center', va='top',
                      weight='bold', fontsize=10, color=color_tit)
        ax_panel.text(0.06, 0.78, cuerpo, va='top', fontsize=9.5,
                      linespacing=1.55, color=ESTILO['color_eje'])

    # ── Tab 3: Ecuaciones ────────────────────────────────
    else:
        # CORRECCIÓN PRINCIPAL: Ajuste de espaciado y cajas para evitar superposición
        ax_panel.text(0.5, 0.97, 'ECUACIONES OPTICAS',
                      ha='center', va='top', weight='bold', fontsize=10.5)
        ax_panel.axhline(0.92, color='#BDC3C7', lw=1, xmin=0.03, xmax=0.97)

        bbox_style = dict(facecolor='#EBF5FB', edgecolor='#AED6F1',
                          lw=1.0, pad=8, boxstyle='round,pad=0.5')

        # 1. Ecuacion de la lente delgada
        y_pos = 0.86
        ax_panel.text(0.05, y_pos, 'Ecuacion de lente delgada:',
                      va='center', fontsize=9, color='#5D6D7E')
        ax_panel.text(0.65, y_pos,
                      r'$\dfrac{1}{f} = \dfrac{1}{s_o} + \dfrac{1}{s_i}$',
                      ha='center', va='center', fontsize=16,
                      bbox=bbox_style)

        # 2. Magnificacion
        y_pos = 0.66
        ax_panel.text(0.05, y_pos, 'Magnificacion lineal:',
                      va='center', fontsize=9, color='#5D6D7E')
        ax_panel.text(0.65, y_pos,
                      r'$M = -\dfrac{s_i}{s_o} = \dfrac{h_i}{h_o}$',
                      ha='center', va='center', fontsize=16,
                      bbox=bbox_style)

        # 3. Despejando si
        y_pos = 0.46
        ax_panel.text(0.05, y_pos, 'Distancia imagen (despejada):',
                      va='center', fontsize=9, color='#5D6D7E')
        ax_panel.text(0.65, y_pos,
                      r'$s_i = \dfrac{f \cdot s_o}{s_o - f}$',
                      ha='center', va='center', fontsize=16,
                      bbox=bbox_style)

        # 4. Convenio de signos
        ax_panel.text(0.05, 0.30, 'Convenio de signos:',
                      va='top', fontsize=9, color='#5D6D7E')
        convenio = (
            "  f > 0  → Convexo (convergente)\n"
            "  f < 0  → Concavo  (divergente)\n"
            "  si > 0 → Imagen real  (proyectable)\n"
            "  si < 0 → Imagen virtual\n"
            "  M < 0  → Imagen invertida\n"
            " |M| > 1 → Imagen aumentada"
        )
        ax_panel.text(0.05, 0.25, convenio, va='top', fontsize=8.5,
                      family='monospace', linespacing=1.4,
                      bbox=dict(facecolor='#F9F9F9', edgecolor='#BDC3C7',
                                lw=0.8, pad=6, boxstyle='round'))

        # Valores actuales
        ax_panel.text(0.50, 0.02,
                      f"Valores actuales:  f={data['f']:.1f}   "
                      f"so={data['so']:.1f}   si={data['si']:.1f} cm",
                      ha='center', va='bottom', fontsize=8, color='#707070')

# ── 7d. Lente adaptable segun f ───────────────────────────
def dibujar_lente(ax, f_val, f_mag, tipo):
    """Lente mas gordo/delgado segun el valor de |f|.
    A menor focal -> mas curvado -> lente mas ancho (convexo) o mas indentado (concavo)."""
    ly = np.linspace(-3.6, 3.6, 120)
    factor = np.clip(ESTILO['f_ref'] / f_mag, 0.35, 2.4)   # escala respecto a f_ref=10

    if tipo == 'Convexo':
        w_centro = 0.75 * factor     # ancho maximo (centro)
        w_borde  = 0.04 * factor     # ancho minimo (extremos)
        xr =  w_borde + (w_centro - w_borde) * (1 - (ly/3.6)**2)
    else:
        w_borde  = 0.65 * factor     # grueso en los bordes
        w_centro = 0.06 * factor     # delgado en el centro
        xr =  w_centro + (w_borde - w_centro) * (ly/3.6)**2

    xl = -xr
    ax.fill_betweenx(ly, xl, xr,
                     color=ESTILO['color_lente'], alpha=ESTILO['alpha_lente'])
    ax.plot(xl, ly, color=ESTILO['color_lente'], lw=1.8)
    ax.plot(xr, ly, color=ESTILO['color_lente'], lw=1.8)
    # Eje central del lente
    ax.plot([0,0], [-3.6, 3.6], color=ESTILO['color_lente'], lw=1.0, alpha=0.4)

# ── 7e. Banco optico principal ────────────────────────────
def dibujar_banco(data, xs, f_mag, tipo_lente):
    so, si, f, M = data['so'], data['si'], data['f'], data['M']
    h_obj = 2.0
    h_img = h_obj * M

    # Limites del eje
    x_max = min(max(si if data['es_real'] else 0, xs) + 18, 120)
    x_min = -so - 20
    ax_bench.set_xlim(x_min, x_max)
    ax_bench.set_ylim(-4.2, 4.5)
    ax_bench.axhline(0, color=ESTILO['color_eje'], lw=1.5, zorder=1)
    ax_bench.grid(True, linestyle='--', alpha=0.25)
    ax_bench.set_title(
        f"Riel Optico  |  Lente {tipo_lente}  |  "
        f"f={f:.1f} cm   so={so:.1f} cm   si={si:.1f} cm   M={M:.2f}",
        weight='bold', fontsize=9, pad=6)

    x_end = min(x_max - 3, max(si if data['es_real'] else 0, xs) + 12)

    # ── Fuente de luz ────────────────────────────────────
    fx = x_min + 5
    ax_bench.add_patch(plt.Rectangle((fx-4, -h_obj*1.2), 4, h_obj*2.4,
                       color='#34495E', zorder=2))
    ax_bench.text(fx-2, h_obj*1.5, 'Fuente\nde Luz', ha='center', fontsize=7.5)

    # Haz de iluminacion difusa (fondo)
    ax_bench.fill_betweenx([-h_obj*1.1, h_obj*1.1], fx, -so,
                            color=ESTILO['color_luz_fuente'], alpha=0.22)

    # ── RAYOS FISICOS DE LA LAMPARA: paralelos, se detienen en el acetato ──
    # Son paralelos porque vienen de una fuente lejana (o con lente colimador).
    # Cada rayo ilumina UN PUNTO del acetato y ahi termina su trayectoria
    # como rayo individual: el acetato absorbe (tinta) o transmite (transparente).
    ys_lamp = np.linspace(-h_obj * 0.88, h_obj * 0.88, 9)
    dx_lamp = (-so) - fx - 0.4       # largo de cada flecha, para antes del acetato
    ax_bench.quiver(
        np.full(len(ys_lamp), fx), ys_lamp,
        np.full(len(ys_lamp), dx_lamp), np.zeros(len(ys_lamp)),
        color='#F1C40F', alpha=0.60, scale=1, scale_units='xy',
        angles='xy', width=0.004, headwidth=4, headlength=5, zorder=3)

    # Etiqueta aclaratoria sobre los rayos fisicos
    ax_bench.text((fx + (-so)) / 2, -h_obj * 1.35,
                  'Rayos fisicos (paralelos)\nse detienen en el acetato',
                  ha='center', va='top', fontsize=6.5,
                  color='#B7950B', style='italic')

    # ── Acetato ───────────────────────────────────────────
    ax_bench.plot([-so,-so], [-h_obj*1.1, h_obj*1.1],
                  color=ESTILO['color_acetato'], lw=4)
    ax_bench.arrow(-so, 0, 0, h_obj, color='black', lw=2,
                   head_width=0.5, length_includes_head=True, zorder=5)
    ax_bench.text(-so, h_obj+0.4, 'Acetato', ha='center', fontsize=8,
                  weight='bold',
                  bbox=dict(facecolor='white', edgecolor='none', alpha=0.75))

    # Indicador de re-emision: cada punto transparente del acetato
    # se convierte en fuente secundaria (Principio de Huygens).
    # De ahi salen los rayos en TODAS las direcciones hacia el lente.
    for yr_h in np.linspace(-h_obj * 0.8, h_obj * 0.8, 5):
        ax_bench.plot(-so, yr_h, 'o', ms=3.5,
                      color='#F39C12', alpha=0.55, zorder=4)
    ax_bench.annotate('', xy=(-so + 1.5, h_obj * 0.55),
                      xytext=(-so, h_obj * 0.55),
                      arrowprops=dict(arrowstyle='->', color='#F39C12',
                                      lw=0.8, alpha=0.5))
    ax_bench.annotate('', xy=(-so + 1.2, h_obj * 0.0),
                      xytext=(-so, h_obj * 0.0),
                      arrowprops=dict(arrowstyle='->', color='#F39C12',
                                      lw=0.8, alpha=0.5))
    ax_bench.annotate('', xy=(-so + 1.5, -h_obj * 0.55),
                      xytext=(-so, -h_obj * 0.55),
                      arrowprops=dict(arrowstyle='->', color='#F39C12',
                                      lw=0.8, alpha=0.5))
    ax_bench.text(-so + 1.8, h_obj * 1.15,
                  'Re-emision\n(Huygens)',
                  ha='left', va='top', fontsize=6,
                  color='#F39C12', style='italic')

    # ── Focos ─────────────────────────────────────────────
    for fx_v, lbl in [(f,'F'), (-f,"F'")]:
        if x_min < fx_v < x_max:
            ax_bench.plot([fx_v, fx_v], [-0.35, 0.35], 'k-', lw=1.5)
            ax_bench.plot(fx_v, 0, 'kx', ms=6)
            ax_bench.text(fx_v, -0.75, lbl, ha='center', fontsize=9)

    # ── Pantalla ──────────────────────────────────────────
    ax_bench.axvline(xs, color=ESTILO['color_pantalla'], lw=5, zorder=2)
    ax_bench.text(xs, 3.5, 'Pantalla', ha='center', fontsize=8, weight='bold',
                  bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))

    # ── Lente ─────────────────────────────────────────────
    dibujar_lente(ax_bench, f, f_mag, tipo_lente)

    # ── Imagen ────────────────────────────────────────────
    if si != np.inf and x_min < si < x_max:
        color_img = ESTILO['color_imagen'] if data['es_real'] else ESTILO['color_virtual']
        ls_img    = '-' if data['es_real'] else '--'
        ax_bench.arrow(si, 0, 0, h_img, color=color_img, lw=2,
                       linestyle=ls_img,
                       head_width=0.5, length_includes_head=True, zorder=5)
        if not data['es_real']:
            ax_bench.text(si, h_img + 0.4 * np.sign(h_img),
                          'Imagen\nvirtual', ha='center', fontsize=7.5,
                          color=ESTILO['color_virtual'],
                          bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # ══════════════════════════════════════════════════════════════
    #  RAYOS DE CONSTRUCCION GEOMETRICA
    #  ─────────────────────────────────────────────────────────────
    #  Los rayos de la lampara (amarillos) son los RAYOS FISICOS:
    #  llegan paralelos al acetato y se detienen ahi.
    #
    #  Por el Principio de Huygens, cada punto transparente del
    #  acetato se convierte en una fuente secundaria que re-emite
    #  luz en TODAS las direcciones hacia el lente.
    #
    #  Los 3 rayos coloreados que se dibujan a continuacion son
    #  RAYOS DE CONSTRUCCION GEOMETRICA: se eligen desde la PUNTA
    #  del objeto porque sus trayectorias son faciles de trazar
    #  sin calculos, y su interseccion revela donde se forma la
    #  imagen de esa punta.  No son los rayos fisicos de la lampara.
    # ══════════════════════════════════════════════════════════════
    if si == np.inf:
        return

    # Separador visual: linea punteada en x = -so para marcar
    # el limite entre "rayos fisicos" y "rayos de construccion"
    ax_bench.axvline(-so, color='#AAB7B8', lw=0.8, ls=':', alpha=0.5, zorder=1)

    # Etiqueta explicativa en la zona de construccion
    mid_x = (-so + 0) / 2
    ax_bench.text(mid_x, ax_bench.get_ylim()[1] * 0.92,
                  'Rayos de construccion geometrica\n'
                  '(salen del punto objeto hacia el lente)',
                  ha='center', va='top', fontsize=6.5,
                  color='#566573', style='italic',
                  bbox=dict(facecolor='white', edgecolor='#AAB7B8',
                            alpha=0.75, pad=2, boxstyle='round'))

    # ── R1: sale PARALELO al eje desde la punta del objeto ────
    #   Despues del lente: se desvía hacia F' (propiedad del lente)
    ax_bench.plot([-so, 0], [h_obj, h_obj],
                  color=ESTILO['color_r1'], lw=1.8, zorder=4,
                  label="R1: Paralelo al eje → pasa por F'")
    y1_end = h_obj * (1 - x_end / f)
    ax_bench.plot([0, x_end], [h_obj, y1_end],
                  color=ESTILO['color_r1'], lw=1.8, zorder=4)

    # ── R2: pasa por el CENTRO OPTICO del lente ───────────────
    #   No se desvía (propiedad del centro del lente)
    slope_c = -h_obj / so
    ax_bench.plot([-so, x_end], [h_obj, slope_c * x_end],
                  color=ESTILO['color_r2'], lw=1.8, zorder=4,
                  label='R2: Por el centro optico (sin desvio)')

    # ── R3: pasa por el FOCO ANTERIOR F antes del lente ──────
    #   Sale paralelo al eje despues del lente
    y3 = None
    if abs(so - f) > 0.3:
        y3 = -h_obj * f / (so - f)
        ax_bench.plot([-so, 0], [h_obj, y3],
                      color=ESTILO['color_r3'], lw=1.8, zorder=4,
                      label='R3: Por F → sale paralelo al eje')
        ax_bench.plot([0, x_end], [y3, y3],
                      color=ESTILO['color_r3'], lw=1.8, zorder=4)

    # ── Marca de interseccion (punto imagen) ─────────────────
    if data['es_real'] and x_min < si < x_max:
        ax_bench.plot(si, h_img, '*', ms=12, color='white', zorder=6,
                      markeredgecolor=ESTILO['color_imagen'], markeredgewidth=1.5)

    # ── Extensiones virtuales punteadas (imagen virtual) ─────
    #   Los rayos DIVERGEN despues del lente; sus prolongaciones
    #   hacia atras convergen en la imagen virtual (si < 0)
    if not data['es_real'] and x_min < si < 0:
        y1_si = h_obj * (1 - si / f)
        y2_si = slope_c * si
        for x_pts, y_pts, col in [
            ([0, si], [h_obj, y1_si], ESTILO['color_r1']),
            ([0, si], [0,     y2_si], ESTILO['color_r2']),
        ]:
            ax_bench.plot(x_pts, y_pts, color=col,
                          lw=1.3, ls='--', alpha=0.55, zorder=3)
        if y3 is not None:
            ax_bench.plot([0, si], [y3, y3],
                          color=ESTILO['color_r3'],
                          lw=1.3, ls='--', alpha=0.55, zorder=3)
        ax_bench.plot(si, h_img, '*', ms=11, color=ESTILO['color_virtual'],
                      zorder=6, markeredgecolor='gray', markeredgewidth=0.8)

    # ── Leyenda ───────────────────────────────────────────────
    from matplotlib.lines import Line2D
    handles_leyenda = [
        Line2D([0],[0], color='#F1C40F', lw=1.5, alpha=0.7,
               label='Rayos fisicos de la lampara (paralelos, se detienen en acetato)'),
        Line2D([0],[0], color=ESTILO['color_r1'], lw=1.8,
               label="R1 (construccion): Paralelo al eje → pasa por F'"),
        Line2D([0],[0], color=ESTILO['color_r2'], lw=1.8,
               label='R2 (construccion): Por el centro optico'),
        Line2D([0],[0], color=ESTILO['color_r3'], lw=1.8,
               label='R3 (construccion): Por F → sale paralelo'),
        Line2D([0],[0], color='gray', lw=1.3, ls='--',
               label='Extension virtual (prolongacion hacia atras)'),
    ]
    ax_bench.legend(handles=handles_leyenda, loc='upper right',
                    fontsize=6.8, framealpha=0.90,
                    ncol=2, handlelength=1.8, labelspacing=0.35)

# ══════════════════════════════════════════════════════════
#  ÁREA 8: UPDATE PRINCIPAL
# ══════════════════════════════════════════════════════════
def update(val=None):
    tipo   = radio_lente.value_selected
    obj    = radio_obj.value_selected
    so_v   = sl_so.val
    f_mag  = sl_f.val
    xs_v   = sl_xs.val

    data, img_proy = calcular_fisica(so_v, f_mag, tipo, xs_v, obj)

    ax_bench.clear()
    dibujar_vista_objeto(obj)
    dibujar_vista_pantalla(img_proy, data)
    dibujar_panel(data, xs_v)
    dibujar_banco(data, xs_v, f_mag, tipo)

    fig.canvas.draw_idle()

# ══════════════════════════════════════════════════════════
#  ÁREA 9: CALLBACKS Y ARRANQUE
# ══════════════════════════════════════════════════════════
def set_tab(nombre):
    def cb(event):
        estado_app['tab'] = nombre
        update()
    return cb

btn_tab1.on_clicked(set_tab('datos'))
btn_tab2.on_clicked(set_tab('analisis'))
btn_tab3.on_clicked(set_tab('ecuaciones'))

radio_lente.on_clicked(update)
radio_obj.on_clicked(update)
sl_so.on_changed(update)
sl_f.on_changed(update)
sl_xs.on_changed(update)

update()

import traceback
try:
    plt.show()
except Exception as e:
    print("ERROR:")
    print(traceback.format_exc())
    input("Enter para salir...")
