import library

# ----------------- Cascades de rosto -----------------

# Frontal
cascade_path_frontal = library.cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade_frontal = library.cv2.CascadeClassifier(cascade_path_frontal)

# Perfil (lado) – só deteta um dos lados, vamos usar flip para o outro
cascade_path_profile = library.cv2.data.haarcascades + "haarcascade_profileface.xml"
face_cascade_profile = library.cv2.CascadeClassifier(cascade_path_profile)

BASE_DIR = library.os.path.dirname(__file__)
DB_PATH = library.os.path.join(BASE_DIR, "dataset")

# Pergunta o nome da pessoa
pessoa = input("Nome da pessoa para guardar as fotos: ").replace(" ", "")

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

janela_nome = "Sorria e vá virando a cabeça 👀"
library.cv2.namedWindow(janela_nome, library.cv2.WINDOW_NORMAL)

print(">>> Vai começar a capturar automaticamente!")
print(">>> Olha para a câmara, depois vira ligeiro para a esquerda e para a direita.")
print(">>> A tirar fotos em 3 segundos...\n")

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

    # -------------- DETEÇÃO DE VÁRIOS ÂNGULOS --------------

    faces_todas = []

    # 1) Faces frontais
    faces_front = face_cascade_frontal.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(120, 120)
    )
    for (x, y, w, h) in faces_front:
        faces_todas.append((x, y, w, h))

    # 2) Faces de perfil (um lado)
    faces_profile = face_cascade_profile.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120)
    )
    for (x, y, w, h) in faces_profile:
        faces_todas.append((x, y, w, h))

    # 3) Faces de perfil no lado contrário (imagem invertida)
    gray_flip = library.cv2.flip(gray, 1)
    faces_profile_flip = face_cascade_profile.detectMultiScale(
        gray_flip,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120)
    )
    for (x_f, y_f, w_f, h_f) in faces_profile_flip:
        # converter coordenadas de volta para o frame normal
        x = w_frame - x_f - w_f
        y = y_f
        w = w_f
        h = h_f
        faces_todas.append((x, y, w, h))

    rosto_valido = None

    # -------------- FILTROS SIMPLES (para evitar lixo) --------------

    for (x, y, w, h) in faces_todas:
        # Filtro de tamanho mínimo (evitar caras muito pequenas ao fundo)
        if w < w_frame * 0.12 or h < h_frame * 0.12:
            continue

        # Filtro de posição demasiado em baixo (tipo peito)
        if y > h_frame * 0.80:
            continue

        # Aqui NÃO fazemos filtro de "aspect ratio",
        # para não excluir caras de lado / inclinadas.

        rosto_valido = (x, y, w, h)
        break

    if rosto_valido:
        x, y, w, h = rosto_valido

        # Quadrado verde
        library.cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        agora = library.time.time()
        if agora - ultimo_tiro >= intervalo:
            nome_arquivo = f"{pessoa}_{contador}.jpg"
            caminho_final = library.os.path.join(pasta_pessoa, nome_arquivo)

            # 🔹 RECORTA SÓ A CARA E REDIMENSIONA (por ex. 224x224)
            rosto = frame[y:y + h, x:x + w]
            rosto = library.cv2.resize(rosto, (224, 224))

            library.cv2.imwrite(caminho_final, rosto)

            print(f"[{contador}/{total_fotos}] Foto guardada: {caminho_final}")
            contador += 1
            ultimo_tiro = agora

    # Texto
    texto = f"{pessoa} - foto {contador}/{total_fotos}"
    library.cv2.putText(
        frame,
        texto,
        (10, 30),
        library.cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    library.cv2.imshow(janela_nome, frame)

    if library.cv2.waitKey(1) & 0xFF == 27:  # ESC
        print(">>> Captura cancelada pelo utilizador.")
        break

print("\n>>> Captura concluída!")
cap.release()
library.cv2.destroyAllWindows()
