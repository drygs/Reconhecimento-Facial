import library
import pickle

BASE_DIR = library.os.path.dirname(__file__)
DB_PATH = library.os.path.join(BASE_DIR, "dataset")
EMB_PATH = library.os.path.join(BASE_DIR, "db_embeddings.pkl")

MODEL_NAME = "Facenet"  # podes testar "VGG-Face", "ArcFace", etc.

# -------------------------------------------------
# Construir base de dados de embeddings (1ª vez)
# -------------------------------------------------
def build_db_embeddings():
    print("🔧 A construir base de dados de embeddings (pode demorar na primeira vez)...")

    db_embeddings = []
    primeira_imagem_path = None  # vamos usar depois para o threshold

    if not library.os.path.exists(DB_PATH):
        print(f"⚠️ Pasta dataset não existe: {DB_PATH}")
        return [], None

    for pessoa in library.os.listdir(DB_PATH):
        pasta_pessoa = library.os.path.join(DB_PATH, pessoa)
        if not library.os.path.isdir(pasta_pessoa):
            continue

        imagens = [
            f for f in library.os.listdir(pasta_pessoa)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not imagens:
            continue

        # guardar a primeira imagem que encontrarmos para calcular o threshold
        if primeira_imagem_path is None:
            primeira_imagem_path = library.os.path.join(pasta_pessoa, imagens[0])

        embeddings_pessoa = []

        total = len(imagens)
        print(f"\n👤 Pessoa: {pessoa} ({total} imagens)")

        for i, nome_img in enumerate(imagens, start=1):
            caminho_img = library.os.path.join(pasta_pessoa, nome_img)
            print(f"  [{i}/{total}] {caminho_img}")

            img = library.cv2.imread(caminho_img)
            if img is None:
                print(f"    ⚠️ Não consegui ler a imagem: {caminho_img}")
                continue

            try:
                reps = library.DeepFace.represent(
                    img_path=img,              # usamos o array da imagem
                    model_name=MODEL_NAME,
                    enforce_detection=False    # já vem recortada, pode estar de lado
                )
            except Exception as e:
                print(f"    ⚠️ Erro a obter embedding: {e}")
                continue

            if not reps or len(reps) == 0:
                print(f"    ⚠️ Não foi possível obter embedding.")
                continue

            emb = library.np.array(reps[0]["embedding"], dtype="float32")
            embeddings_pessoa.append(emb)

        if not embeddings_pessoa:
            print("  ⚠️ Nenhuma embedding válida para esta pessoa, a saltar.")
            continue

        # média das embeddings dessa pessoa → usa TODAS as fotos (frente + lado)
        mean_emb = library.np.mean(embeddings_pessoa, axis=0)

        db_embeddings.append({
            "pessoa": pessoa,
            "embedding": mean_emb,
        })

    if not db_embeddings:
        print("⚠️ Não foram encontradas imagens válidas em dataset/<pessoa>/")
        return [], None

    # guardar em ficheiro para arrancar rápido da próxima vez
    try:
        with open(EMB_PATH, "wb") as f:
            pickle.dump({
                "db_embeddings": db_embeddings,
                "sample_image": primeira_imagem_path
            }, f)
        print(f"\n✅ Base de dados guardada em: {EMB_PATH}")
    except Exception as e:
        print(f"\n⚠️ Erro a guardar base de dados: {e}")

    return db_embeddings, primeira_imagem_path

# -------------------------------------------------
# Carregar base de dados (ou construir se não existir)
# -------------------------------------------------
def load_db_embeddings():
    if library.os.path.exists(EMB_PATH):
        try:
            with open(EMB_PATH, "rb") as f:
                data = pickle.load(f)
            db = data.get("db_embeddings", [])
            sample_image = data.get("sample_image", None)

            if db and sample_image and library.os.path.exists(sample_image):
                print(f"✅ Base de dados carregada de {EMB_PATH}")
                return db, sample_image
            else:
                print("⚠️ Ficheiro de embeddings incompleto, vou reconstruir...")
        except Exception as e:
            print(f"⚠️ Erro a carregar {EMB_PATH}: {e}. Vou reconstruir...")

    # Se chegou aqui, é porque não havia ficheiro ou deu erro → reconstruir
    return build_db_embeddings()

# -------------------------------------------------
# Carregar base de dados de rostos
# -------------------------------------------------
print("A carregar base de dados de rostos...")
db_embeddings, sample_image_path = load_db_embeddings()

if not db_embeddings:
    print("⚠️ Base de dados de rostos vazia. A sair.")
    exit()

print("Pessoas encontradas na base de dados:")
for item in db_embeddings:
    print(" -", item["pessoa"])

# -------------------------------------------------
# Limiar (threshold) do modelo
# -------------------------------------------------
print("A carregar modelo de reconhecimento (só a 1ª vez demora um pouco)...")

if sample_image_path is None or not library.os.path.exists(sample_image_path):
    print("⚠️ Não há imagem sample para calcular threshold.")
    exit()

img_primeira = library.cv2.imread(sample_image_path)

ref = library.DeepFace.verify(
    img1_path=img_primeira,
    img2_path=img_primeira,
    model_name=MODEL_NAME,
    enforce_detection=False
)

base_threshold = float(ref["threshold"])
# Para caras de lado, deixamos o limiar mais folgado
LIMIAR = base_threshold * 1.3

print(f"Modelo pronto! Threshold base: {base_threshold:.4f} | Limiar usado: {LIMIAR:.4f}")

# -------------------------------------------------
# Webcam
# -------------------------------------------------
cap = library.cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro a abrir webcam")
    exit()

print(">>> Webcam ligada. Aproxima a cara (também pode ser de lado). ESC para sair.")

FRAME_SKIP = 10  # processa 1 em cada 3 frames
frame_index = 0

# último resultado detetado (para desenhar também nos frames em que não processa)
ultimas_faces = []  # lista de dicts: { "nome": ..., "dist": ..., "box": (x,y,w,h) }

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1

    # Reduz um pouco a imagem para acelerar DeepFace
    frame_pequeno = library.cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)

    h_full, w_full = frame.shape[:2]
    h_small, w_small = frame_pequeno.shape[:2]
    sx = w_full / float(w_small)
    sy = h_full / float(h_small)

    # Só processa em alguns frames para ficar mais fluido
    processar_este = (frame_index % FRAME_SKIP == 0)

    if processar_este:
        faces_detectadas = []

        try:
            reps_frame = library.DeepFace.represent(
                img_path=frame_pequeno,
                model_name=MODEL_NAME,
                enforce_detection=False   # deixa passar mesmo se estiver meio de lado
            )

            if reps_frame is None:
                reps_frame = []

            # algumas versões devolvem dict, outras lista
            if isinstance(reps_frame, dict):
                reps_lista = [reps_frame]
            else:
                reps_lista = reps_frame

            for rep in reps_lista:
                if "embedding" not in rep:
                    continue

                emb_frame = library.np.array(rep["embedding"], dtype="float32")

                # tentar apanhar a área da cara
                facial_area = rep.get("facial_area") or rep.get("region")
                if facial_area and isinstance(facial_area, dict):
                    x = facial_area.get("x", 0)
                    y = facial_area.get("y", 0)
                    w = facial_area.get("w", 0)
                    h = facial_area.get("h", 0)
                else:
                    # se não houver coords, não conseguimos desenhar caixa
                    x = y = w = h = 0

                melhor_dist = None
                melhor_nome = "Desconhecido"

                # compara com todas as embeddings da BD (cosine distance)
                for item in db_embeddings:
                    emb_db = item["embedding"]

                    dot = float(library.np.dot(emb_frame, emb_db))
                    norm_prod = float(
                        library.np.linalg.norm(emb_frame) * library.np.linalg.norm(emb_db)
                    ) + 1e-8
                    cos_sim = dot / norm_prod
                    dist = 1.0 - cos_sim

                    if melhor_dist is None or dist < melhor_dist:
                        melhor_dist = dist
                        melhor_nome = item["pessoa"]

                if melhor_dist is not None and melhor_dist < LIMIAR:
                    label = f"{melhor_nome} ({melhor_dist:.3f})"
                else:
                    label = "Desconhecido"

                # guardar para depois desenhar
                faces_detectadas.append({
                    "nome": label,
                    "dist": melhor_dist,
                    "box_small": (x, y, w, h),
                })

        except Exception as e:
            # se der erro, não atualiza deteções neste frame
            # print("[ERRO represent]:", e)
            faces_detectadas = []

        # converter coordenadas do frame pequeno para o frame original
        ultimas_faces = []
        for f in faces_detectadas:
            x_s, y_s, w_s, h_s = f["box_small"]
            # escala para o tamanho original
            x_f = int(x_s * sx)
            y_f = int(y_s * sy)
            w_f = int(w_s * sx)
            h_f = int(h_s * sy)

            ultimas_faces.append({
                "nome": f["nome"],
                "dist": f["dist"],
                "box": (x_f, y_f, w_f, h_f),
            })

    # desenhar sempre as últimas caras detetadas (mesmo em frames não processados)
    for f in ultimas_faces:
        x, y, w, h = f["box"]
        if w > 0 and h > 0:
            library.cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            library.cv2.putText(
                frame,
                f["nome"],
                (x, max(0, y - 10)),
                library.cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    library.cv2.imshow("Reconhecimento Facial", frame)

    if library.cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
library.cv2.destroyAllWindows()
