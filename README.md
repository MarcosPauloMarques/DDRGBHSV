DDRGBHSV (Detector de Desmatamento utilizando RGB e HSV).

Classifica cada pixel e pinta o resultado:
    VERMELHO(255, 0, 0) -> desmatamento (solo exposto, pasto, áreas abertas)
    VERDE(0, 170, 0)    -> floresta
    BRANCO(255, 255, 255)   -> nuvens

Uso:
    python detectar_desmatamento.py imagem.jpg
    python detectar_desmatamento.py pasta_com_imagens/
    python detectar_desmatamento.py a.jpg b.png "cenas/*.tif"

Comando para instalar Dependências :
    pip install numpy pillow scipy matplotlib rasterio

Notas:
	A pasta "ImagensOriginais" contém um conjunto de 45 imagens que foram utilizadas para testes e validação e podem ser reutilizadas como entrada (input).
	A pasta "saída" contém o Output das imagens de Input.
