import library

BASE_DIR = library.os.path.dirname(__file__)
DB_PATH = library.os.path.join(BASE_DIR, "dataset")

# Modelo que o DeepFace vai usar (podes trocar por "Facenet", "ArcFace", etc.)
MODEL_NAME = "Facenet"  # ou "VGG-Face", "ArcFace", "SFace", etc.

# Opcional: força o DeepFace a construir a DB logo no início (melhor performance)
print("A preparar base de dados de rostos...")
library.DeepFace.find(img_path=library.os.path.join(DB_PATH, library.os.listdir(library.os.path.join(DB_PATH, library.os.listdir(DB_PATH)[0]))[0]),
              db_path=DB_PATH,
              model_name=MODEL_NAME,
              enforce_detection=False)
print("Base de dados pronta!")

cap = library.cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro a abrir webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # DeepFace espera BGR ou RGB consoante a versão; aqui deixamos BGR e ele trata
    try:
        # Faz a procura na DB
        dfs = library.DeepFace.find(
            img_path=frame,
            db_path=DB_PATH,
            model_name=MODEL_NAME,
            enforce_detection=False
        )

        nome_mostrar = "Desconhecido"

        if len(dfs) > 0:
            df = dfs[0]  # primeiro DataFrame (um por modelo)
            if not df.empty:
                # Primeiro match
                best_row = df.iloc[0]
                caminho_img = best_row["identity"]
                distancia = best_row["distance"]

                # Nome = nome da pasta (pessoa)
                pessoa = library.os.path.basename(library.os.path.dirname(caminho_img))

                # Define um limiar de distância (ajustar conforme o modelo)
                LIMIAR = 0.7  # quanto menor, mais parecido. Ajusta se necessário.

                if distancia < LIMIAR:
                    nome_mostrar = f"{pessoa} ({distancia:.2f})"

        # Mostra o nome no ecrã
        library.cv2.putText(frame, nome_mostrar, (20, 40),
                    library.cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        # Se não encontrar rosto ou der outro erro, ignora e mostra só a imagem
        # print("Erro:", e)
        pass

    library.cv2.imshow("Reconhecimento Facial - Juca IA", frame)

    if library.cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

library.cap.release()
library.cv2.destroyAllWindows()
