Agent Workflow:

Fetch Tasks → Get all tasks from contract
Process Proofs → Find tasks with submitted proof images
Analyze Images → BLIP captioning + Jenius analysis
Verify Completion → Embeddings + LLM decision making
Execute Action → Approve completion or mark failed
Check Expired → Mark overdue tasks as failed
Log Results → Track everything in Opik


1. BLIP image captioning: "screenshot of code editor with smart contract"
2. Sentence similarity: task_desc vs image_desc (0.85 similarity)
3. LLM verification: "Does this image show task completion?"
4. Decision: COMPLETE → approve task + mint NFT

llamaindex github for validating github tasks instead of image tasks


Quick Start:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup Ollama + LLaMA3
ollama serve
ollama pull llama3

# 3. Set environment variables
export PRIVATE_KEY="your_key"
export JENIUS_TOKEN="your_token"
OPIK_API_KEY="your_key"
OPIK_WORKSPACE="your_workspace_name"

# 4. Activate venv
source venv/bin/activate

# 5. Run the agent
python taskfi_agent.py
```