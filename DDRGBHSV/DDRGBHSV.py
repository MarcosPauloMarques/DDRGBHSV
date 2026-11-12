#!/usr/bin/env python3

import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None

# --------------------------------------------------------------------------- #
# Parâmetros fixos da verificação
# --------------------------------------------------------------------------- #
PASTA_SAIDA = "saida"
EXTENSOES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

NUVEM_V = 0.70       # brilho mínimo de nuvem (0-1)
NUVEM_S = 0.22       # saturação máxima de nuvem (0-1)
RG_MAX = 0.68        # razão vermelho/verde máxima para ser floresta
FLORESTA_V = 0.45    # brilho máximo de floresta (0-1)
SIGMA = 1.5          # suavização gaussiana inicial
JANELA = 7           # janela do filtro de moda (px)
BORDA_NUVEM = 15     # faixa (px) ao redor das nuvens tratada como névoa
AREA_MIN = 300       # manchas menores que isso (px) são absorvidas pela vizinhança
TEXTURA_MIN = 0.5    # Sobel: fração da textura típica da floresta abaixo da qual vira desmatamento
JANELA_SOBEL = 15    # janela (px) em que a textura de Sobel é média

# ids das classes
FLORESTA, DESMATAMENTO, NUVEM = 0, 1, 2
NOMES = ["Floresta", "Desmatamento", "Nuvem"]
PALETA = np.array(
    [
        [0, 170, 0],      # floresta     -> verde
        [255, 0, 0],      # desmatamento -> vermelho
        [255, 255, 255],  # nuvem        -> branco
    ],
    dtype=np.uint8,
)


# --------------------------------------------------------------------------- #
# Entrada / saída
# --------------------------------------------------------------------------- #
def carregar_imagem(caminho):
    """Lê a imagem e devolve um array float32 (H, W, 3) em [0, 1]."""
    try:
        img = Image.open(caminho).convert("RGB")
        return np.asarray(img, dtype=np.float32) / 255.0
    except Exception as erro_pil:
        try:
            import rasterio  # opcional

            with rasterio.open(caminho) as src:
                bandas = src.read([1, 2, 3]).astype(np.float32)
            rgb = np.moveaxis(bandas, 0, -1)
            lo, hi = np.percentile(rgb, (2, 98))  # estica o contraste
            return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
        except Exception:
            raise RuntimeError(f"não consegui abrir '{caminho}': {erro_pil}")


# --------------------------------------------------------------------------- #
# Classificação
# --------------------------------------------------------------------------- #
def classificar(rgb):
    """Devolve um mapa (H, W) de ids de classe."""
    x = ndi.gaussian_filter(rgb.astype(np.float32), sigma=(SIGMA, SIGMA, 0))  # reduz ruído

    r, g = x[..., 0], x[..., 1]
    v = x.max(axis=-1)                                   # brilho
    s = (v - x.min(axis=-1)) / (v + 1e-6)                # saturação
    rg = r / (g + 1e-6)                                  # vermelho / verde ("vermelhidão" relativa)

    # 1) nuvens: muito claras e quase sem cor
    nuvem = (v > NUVEM_V) & (s < NUVEM_S)

    # 2) floresta: copa densa = dominada pelo VERDE (R/G baixo) e escura.
    #    Solo exposto, pasto, queimada e áreas abertas têm R/G alto (>= ~0.75).
    floresta = ~nuvem & (rg < RG_MAX) & (v < FLORESTA_V)

    # 2b) textura (Sobel): copa de floresta é rugosa, pasto/mato rasteiro/solo são lisos.
    #     Entre os pixels que passaram no teste de cor, quem tem textura bem abaixo da
    #     típica DA PRÓPRIA IMAGEM (mediana da floresta) é tratado como área aberta.
    if floresta.any():
        tex = mapa_textura(rgb)
        ref = np.median(tex[floresta])
        floresta &= tex >= TEXTURA_MIN * ref

    # 3) todo o resto = solo exposto / pasto / área aberta -> desmatamento
    cls = np.full(r.shape, DESMATAMENTO, dtype=np.uint8)
    cls[floresta] = FLORESTA
    cls[nuvem] = NUVEM

    cls = filtro_moda(cls, JANELA)
    cls = limpar_pequenas(cls, AREA_MIN)

    # 4) bordas de nuvem: a névoa translúcida mistura nuvem + floresta e ficaria
    #    parecida com solo exposto. "Desmatamento" colado numa nuvem vira nuvem.
    if (cls == NUVEM).any():
        dist = ndi.distance_transform_edt(cls != NUVEM)
        cls[(dist <= BORDA_NUVEM) & (cls == DESMATAMENTO)] = NUVEM
    return cls


def sobel_magnitude(img):
    """Magnitude do gradiente pelo filtro de Sobel (kernel normalizado por 8)."""
    gx = ndi.sobel(img, axis=1)
    gy = ndi.sobel(img, axis=0)
    return np.hypot(gx, gy) / 8.0


def mapa_textura(rgb):
    """Textura = média local da magnitude de Sobel da luminância em escala logarítmica
    (o log torna a medida independente do brilho: sombra e sol dão o mesmo contraste)."""
    lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lum = ndi.gaussian_filter(np.log(lum + 0.02), 1.0)
    return ndi.uniform_filter(sobel_magnitude(lum), size=JANELA_SOBEL)


def limpar_pequenas(cls, area_min):
    """Remove manchas pequenas: 'desmatamento' menor que area_min px dentro da floresta vira
    floresta (ruído de copa); 'floresta' menor que area_min px dentro do desmatamento vira
    desmatamento (árvores isoladas no pasto)."""
    for alvo, troca in ((DESMATAMENTO, FLORESTA), (FLORESTA, DESMATAMENTO)):
        mascara = cls == alvo
        rotulos, n = ndi.label(mascara)
        if n:
            tamanhos = ndi.sum(mascara, rotulos, np.arange(1, n + 1))
            pequenas = np.isin(rotulos, 1 + np.flatnonzero(tamanhos < area_min))
            cls[pequenas] = troca
    return cls


def filtro_moda(cls, janela):
    """Cada pixel assume a classe mais comum da vizinhança (remove ruído 'sal e pimenta')."""
    votos = np.stack(
        [ndi.uniform_filter((cls == k).astype(np.float32), size=janela) for k in range(len(NOMES))]
    )
    return votos.argmax(axis=0).astype(np.uint8)


def colorir(cls):
    return PALETA[cls]


def sobrepor(rgb, cls, alfa=0.55):
    base = (rgb * 255).astype(np.float32)
    cor = PALETA[cls].astype(np.float32)
    return np.clip((1 - alfa) * base + alfa * cor, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Relatório
# --------------------------------------------------------------------------- #
def relatorio(cls):
    total = cls.size
    print("\nResultado da classificação")
    print("-" * 30)
    for k, nome in enumerate(NOMES):
        print(f"{nome:<14}{100 * (cls == k).sum() / total:6.2f} %")
    util = total - int((cls == NUVEM).sum())
    if util:
        d = int((cls == DESMATAMENTO).sum())
        print("-" * 30)
        print(f"Desmatamento sobre a área visível (sem nuvens): {100 * d / util:.2f} %")


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #
def listar_imagens(entradas):
    """Expande pastas, arquivos e curingas em uma lista de imagens (sem repetições)."""
    achados = []
    for e in entradas:
        p = Path(e)
        if p.is_dir():
            achados += sorted(f for f in p.glob("*") if f.is_file() and f.suffix.lower() in EXTENSOES)
        elif p.is_file():
            achados.append(p)
        else:  # curingas (o Windows não expande sozinho)
            g = [Path(x) for x in sorted(glob.glob(e))]
            g = [f for f in g if f.is_file() and f.suffix.lower() in EXTENSOES]
            if not g:
                print(f"Aviso: nada encontrado para '{e}'")
            achados += g
    vistos, unicos = set(), []
    for f in achados:
        k = f.resolve()
        if k not in vistos:
            vistos.add(k)
            unicos.append(f)
    return unicos


def processar(rgb, nome, pasta):
    """Classifica uma imagem, salva as saídas e imprime o relatório."""
    cls = classificar(rgb)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(pasta / f"{nome}_original.png")
    Image.fromarray(colorir(cls)).save(pasta / f"{nome}_classificado.png")
    Image.fromarray(sobrepor(rgb, cls)).save(pasta / f"{nome}_sobreposicao.png")
    relatorio(cls)


def main():
    entradas = sys.argv[1:]
    if not entradas:
        sys.exit("Informe a imagem ou a pasta. Exemplo: python detectar_desmatamento.py imagem.jpg")

    pasta = Path(PASTA_SAIDA)
    pasta.mkdir(parents=True, exist_ok=True)

    arquivos = listar_imagens(entradas)
    if not arquivos:
        sys.exit("Nenhuma imagem encontrada (extensões: " + ", ".join(sorted(EXTENSOES)) + ")")

    ok, falhas, usados = 0, [], {}
    for i, arq in enumerate(arquivos, 1):
        print(f"\n=== [{i}/{len(arquivos)}] {arq} ===")
        nome = arq.stem
        usados[nome] = usados.get(nome, 0) + 1
        if usados[nome] > 1:  # mesmo nome em pastas diferentes
            nome = f"{nome}_{usados[nome]}"
        try:
            processar(carregar_imagem(arq), nome, pasta)
            ok += 1
        except Exception as erro:
            print(f"  ERRO: {erro} -> arquivo ignorado")
            falhas.append(str(arq))

    if ok > 1 or falhas:
        print(f"\nProcessadas: {ok}   Falhas: {len(falhas)}")
        for f in falhas:
            print("  falhou:", f)
    print(f"\nArquivos salvos em: {pasta.resolve()}")


if __name__ == "__main__":
    main()