import library

# Carregar Haar Cascade (detetor de rostos)


cascade_path = library.cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = library.cv2.CascadeClassifier(cascade_path)

BASE_DIR = library.os.path.dirname(__file__)
DB_PATH = library.os.path.join(BASE_DIR, "dataset")

# Pergunta o nome da pessoa
pessoa = input("Nome da pessoa para guardar as fotos: ").replace(' ', '')

# Caminho da pasta dessa pessoa
pasta_pessoa = library.os.path.join(DB_PATH, pessoa)

# Cria a pasta se não existir
library.os.makedirs(pasta_pessoa, exist_ok=True)
print(f"Pasta criada / usada: {pasta_pessoa}")

cap = library.cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro a abrir webcam")
    exit()

# Resolução mais leve (podes pôr 1280x720 se quiseres)
cap.set(library.cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(library.cv2.CAP_PROP_FRAME_HEIGHT, 480)

janela_nome = "Sorria"
library.cv2.namedWindow(janela_nome, library.cv2.WINDOW_NORMAL)

print(">>> Vai começar a capturar automaticamente!")
print(">>> Prepara a cara! A tirar fotos em 3 segundos...\n")

library.time.sleep(1); print("3...")
library.time.sleep(1); print("2...")
library.time.sleep(1); print("1...")
print("A capturar 100 fotos automaticamente!\n")

total_fotos = 100
intervalo = 10 / total_fotos  # 100 fotos em 10 segundos
contador = 1
ultimo_tiro = library.time.time()

while contador <= total_fotos:
    ret, frame = cap.read()
    if not ret:
        break

    h_frame, w_frame = frame.shape[:2]
    gray = library.cv2.cvtColor(frame, library.cv2.COLOR_BGR2GRAY)

    # 🔍 detecção melhorada
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,     # mais preciso
        minNeighbors=7,      # evita falsos positivos
        minSize=(120, 120)   # ignora caras pequenas demais
    )

    rosto_valido = None

    for (x, y, w, h) in faces:
        aspect = w / float(h)

        # Filtros anti-queixo
        if aspect < 0.7 or aspect > 1.5:
            continue
        if y > h_frame * 0.55:  # rosto muito em baixo
            continue
        if w < w_frame * 0.15 or h < h_frame * 0.15:
            continue

        rosto_valido = (x, y, w, h)
        break

    if rosto_valido:
        x, y, w, h = rosto_valido

        # Quadrado verde
        library.cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Tira a foto do FRAME INTEIRO
        agora = library.time.time()
        if agora - ultimo_tiro >= intervalo:
            nome_arquivo = f"{pessoa}_{contador}.jpg"
            caminho_final = library.os.path.join(pasta_pessoa, nome_arquivo)
            library.cv2.imwrite(caminho_final, frame)

            print(f"[{contador}/{total_fotos}] Foto guardada: {caminho_final}")
            contador += 1
            ultimo_tiro = agora

    # Texto
    texto = f"{pessoa} - foto {contador}/{total_fotos}"
    library.cv2.putText(frame, texto, (10, 30),
                library.cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    library.cv2.imshow(janela_nome, frame)

    if library.cv2.waitKey(1) & 0xFF == 27:  # ESC
        print(">>> Captura cancelada pelo utilizador.")
        break

print("\n>>> Captura concluída!")
cap.release()
library.cv2.destroyAllWindows()
