import boto3
import json
import psycopg2
from langchain_community.vectorstores import PGVector
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_core.documents import Document

# ==========================================
# CONFIGURATION
# ==========================================

REGION = "ap-south-1"
BEDROCK_TEXT_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"

CONNECTION_STRING = "postgresql+psycopg2://postgres:Shri%402005@localhost:5432/pharmacy_db"

# AWS Clients
textract = boto3.client("textract", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

# LangChain Embedding Wrapper
embeddings = BedrockEmbeddings(
    model_id=EMBEDDING_MODEL_ID,
    region_name=REGION
)

# PGVector Store
vector_store = PGVector(
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
    collection_name="medicine_collection",
)

# ==========================================
# STEP 1: Extract Text from Prescription
# ==========================================

def extract_text_from_image(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = textract.detect_document_text(
        Document={'Bytes': image_bytes}
    )

    lines = [
        block["Text"]
        for block in response["Blocks"]
        if block["BlockType"] == "LINE"
    ]

    return "\n".join(lines)


# ==========================================
# STEP 2: Extract Medicines Using Claude
# ==========================================

def extract_medicines(text):

    prompt = f"""
    Extract only medicine names with dosage from this prescription.
    Return only comma separated values.
    No extra text.

    Text:
    {text}
    """

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
    }

    response = bedrock.invoke_model(
        modelId=BEDROCK_TEXT_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )

    result = json.loads(response["body"].read())
    output = result["content"][0]["text"]

    return [m.strip().lower() for m in output.split(",")]


# ==========================================
# STEP 3: Store Medicines in PGVector
# ==========================================

def store_product_embeddings():

    conn = psycopg2.connect(
        dbname="pharmacy_db",
        user="postgres",
        password="Shri@2005",
        host="localhost",
        port="5432"
    )

    cur = conn.cursor()
    cur.execute("SELECT product_name FROM medicines")
    rows = cur.fetchall()
    conn.close()

    total = len(rows)
    print(f"\n🚀 Starting embedding of {total} medicines...\n")

    for idx, row in enumerate(rows, start=1):

        name = row[0].strip().lower()

        doc = Document(
            page_content=name,
            metadata={"product_name": name}
        )

        vector_store.add_documents([doc])   # Add one at a time

        print(f"Embedding {idx}/{total} → {name}")

    print("\n✅ All embeddings stored successfully!")
# ==========================================
# STEP 4: Similarity Search from PG
# ==========================================
def check_availability(extracted_medicines, top_k=3):

    results = []

    for med in extracted_medicines:

        print(f"\n🔎 Searching for: {med}")

        docs_and_scores = vector_store.similarity_search_with_score(
            med,
            k=top_k
        )

        if not docs_and_scores:
            results.append(f"❌ {med} → NOT AVAILABLE")
            continue

        # Convert distance → similarity score
        top_matches = []
        for doc, score in docs_and_scores:
            similarity = 1 - score  # convert distance to similarity
            top_matches.append((doc.page_content, similarity))

        # Call LLM for final decision
        decision = llm_decide_availability(med, top_matches)

        results.append(f"\n🔹 Requested: {med}\n{decision}")

    return "\n".join(results)

def llm_decide_availability(medicine_name, top_matches):
    print(top_matches)

    matches_text = "\n".join(
        [f"- {name} (score: {round(score,3)})"
         for name, score in top_matches]
    )

    prompt = f"""
    You are a pharmacy assistant.

    Requested Medicine:
    {medicine_name}

    Top Similar Products from Database:
    {matches_text}

    Decide clearly:
    1) AVAILABLE
    2) NOT AVAILABLE

    If available:
    - Mention best match
    - Mention alternatives
    - Be clear and structured

    If not:
    - Say NOT AVAILABLE clearly.
    """

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
    }

    response = bedrock.invoke_model(
        modelId=BEDROCK_TEXT_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"].strip()


# ==========================================
# MAIN
# ==========================================
# ==========================================
# STEP 0: Get User Input (Image or Text)
# ==========================================

def get_user_input():

    choice = input("""
Choose input type:
1 → Upload Prescription Image
2 → Type Medicine Name
Enter 1 or 2: 
""")

    if choice == "1":
        image_path = input("Enter image path: ")
        print("🔎 Extracting prescription text...")
        text = extract_text_from_image(image_path)

        print("\nExtracted Text:\n", text)

        print("\n💊 Extracting medicines using LLM...")
        medicines = extract_medicines(text)

    elif choice == "2":
        typed_input = input("Enter medicine names (comma separated): ")
        medicines = [
            med.strip().lower()
            for med in typed_input.split(",")
            if med.strip()
        ]

    else:
        print("❌ Invalid choice")
        return []

    return medicines

def main():

    medicines = get_user_input()

    if not medicines:
        print("No medicines provided.")
        return

    print("\nMedicines Found:", medicines)

    # ⚠️ Run only first time or when medicines table changes
    # store_product_embeddings()

    print("\n📦 Checking availability from PostgreSQL...")
    availability = check_availability(medicines)

    print("\n========== FINAL RESULT ==========")
    print(availability)


if __name__ == "__main__":
    main()