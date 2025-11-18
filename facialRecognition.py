import library



BASE_DIR = library.os.path.dirname(__file__)
DB_PATH = library.os.path.join(BASE_DIR, "dataset")
EMB_PATH = library.os.path.join(BASE_DIR, "db_embeddings.pkl")

MODEL_NAME = "ArcFace"   # Podes usar: "SFace" (3x mais rápido), "ArcFace", etc.


# -------------------------------------------------
# Função que processa 1 imagem num processo separado
# -------------------------------------------------
def process_single_image(args):
    caminho_img, model = args
    try:
        img = library.cv2.imread(caminho_img)
        if img is None:
            return None

        reps = library.DeepFace.represent(
            img_path=img,
            model_name=model,
            enforce_detection=False
        )

        if not reps:
            return None

        return library.np.array(reps[0]["embedding"], dtype="float32")
    except:
        return None


# -------------------------------------------------
# Criar o ficheiro db_embeddings.pkl usando multiprocessing
# -------------------------------------------------
def build_db_embeddings():
    print("🔧 A gerar db_embeddings.pkl (primeira vez demora um pouco)…")

    db_embeddings = []
    sample_image_path = None

    if not library.os.path.exists(DB_PATH):
        print("⚠️ dataset/ não existe")
        return [], None

    # quantidade de processos
    NUM_CORES = max(1, library.cpu_count() - 1)
    print(f"💻 A usar {NUM_CORES} processos\n")

    for pessoa in library.os.listdir(DB_PATH):
        pasta = library.os.path.join(DB_PATH, pessoa)
        if not library.os.path.isdir(pasta):
            continue

        imagens = [
            f for f in library.os.listdir(pasta)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not imagens:
            continue

        if sample_image_path is None:
            sample_image_path = library.os.path.join(pasta, imagens[0])

        caminhos = [(library.os.path.join(pasta, img), MODEL_NAME) for img in imagens]

        print(f"👤 {pessoa} — {len(imagens)} imagens (processando…)")

        # multiprocessing
        with library.Pool(NUM_CORES) as pool:
            resultados = pool.map(process_single_image, caminhos)

        embeddings = [emb for emb in resultados if emb is not None]

        print(f"   ✔️ {len(embeddings)}/{len(imagens)} usadas\n")

        if not embeddings:
            continue

        mean_emb = library.np.mean(embeddings, axis=0)

        db_embeddings.append({
            "pessoa": pessoa,
            "embedding": mean_emb,
        })

    # guardar pkl
    data = {
        "db_embeddings": db_embeddings,
        "sample_image": sample_image_path
    }

    with open(EMB_PATH, "wb") as f:
        library.pickle.dump(data, f)

    print("✅ db_embeddings.pkl criado!")
    return db_embeddings, sample_image_path


# -------------------------------------------------
# Carregar o .pkl ou criar se não existir
# -------------------------------------------------
def load_db_embeddings():
    if library.os.path.exists(EMB_PATH):
        try:
            with open(EMB_PATH, "rb") as f:
                data = library.pickle.load(f)

            if "db_embeddings" in data and "sample_image" in data:
                print(f"🔌 Base carregada de {EMB_PATH}")
                return data["db_embeddings"], data["sample_image"]
        except:
            print("⚠️ Erro ao carregar PKL, a reconstruir…")

    return build_db_embeddings()


# -------------------------------------------------
# Carregar base de dados
# -------------------------------------------------
print("A carregar base de dados…")
db_embeddings, sample_image_path = load_db_embeddings()

if not db_embeddings:
    print("⚠️ Erro: base de dados vazia.")
    exit()

print("\nPessoas na base de dados:")
for item in db_embeddings:
    print(" -", item["pessoa"])


# -------------------------------------------------
# Limiar do modelo
# -------------------------------------------------
img_sample = library.cv2.imread(sample_image_path)

ref = library.DeepFace.verify(
    img1_path=img_sample,
    img2_path=img_sample,
    model_name=MODEL_NAME,
    enforce_detection=False
)

base_threshold = float(ref["threshold"])
LIMIAR = base_threshold * 1.0

print(f"\nModelo pronto ✔️  Limiar: {LIMIAR:.4f}\n")


# -------------------------------------------------
# Webcam
# -------------------------------------------------
cap = library.cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro a abrir webcam")
    exit()

print(">>> Webcam ligada — aproxima a cara (frente ou de lado). ESC para sair.")

FRAME_SKIP = 6
frame_index = 0

ultimas_faces = []


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1

    frame_small = library.cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)

    h_full, w_full = frame.shape[:2]
    h_small, w_small = frame_small.shape[:2]
    sx = w_full / w_small
    sy = h_full / h_small

    if frame_index % FRAME_SKIP == 0:
        faces_detectadas = []

        try:
            reps = library.DeepFace.represent(
                img_path=frame_small,
                model_name=MODEL_NAME,
                enforce_detection=False
            )

            if isinstance(reps, dict):
                reps = [reps]
            if reps is None:
                reps = []

            for rep in reps:
                if "embedding" not in rep:
                    continue

                emb_frame = library.np.array(rep["embedding"], dtype="float32")

                region = rep.get("region") or rep.get("facial_area") or {}
                x = int(region.get("x", 0) * sx)
                y = int(region.get("y", 0) * sy)
                w = int(region.get("w", 0) * sx)
                h = int(region.get("h", 0) * sy)

                melhor_dist = 10
                melhor_nome = "Desconhecido"

                for item in db_embeddings:
                    emb_db = item["embedding"]
                    dot = float(library.np.dot(emb_frame, emb_db))
                    norm = float(library.np.linalg.norm(emb_frame) *
                                 library.np.linalg.norm(emb_db)) + 1e-8
                    dist = 1 - (dot / norm)

                    if dist < melhor_dist:
                        melhor_dist = dist
                        melhor_nome = item["pessoa"]

                if melhor_dist < LIMIAR:
                    label = f"{melhor_nome} ({melhor_dist:.3f})"
                else:
                    label = "Desconhecido"

                faces_detectadas.append({
                    "nome": label,
                    "box": (x, y, w, h)
                })

        except:
            faces_detectadas = []

        ultimas_faces = faces_detectadas

    for f in ultimas_faces:
        x, y, w, h = f["box"]
        library.cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        library.cv2.putText(frame, f["nome"], (x, y - 10),
                            library.cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)

    library.cv2.imshow("Reconhecimento Facial", frame)

    if library.cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
library.cv2.destroyAllWindows()
