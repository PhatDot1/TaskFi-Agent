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
