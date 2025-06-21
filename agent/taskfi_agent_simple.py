#!/usr/bin/env python3
"""
Simplified TaskFi Monitoring Agent
Compatible version with better error handling and simpler dependencies
"""

import os
import asyncio
import time
import json
import requests
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# Core dependencies
try:
    import torch
    from PIL import Image
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    print("Warning: BLIP dependencies not available. Install transformers and torch.")
    BLIP_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    print("Warning: Sentence transformers not available. Install sentence-transformers.")
    EMBEDDINGS_AVAILABLE = False

try:
    from web3 import Web3
    WEB3_AVAILABLE = True
except ImportError:
    print("Warning: Web3 not available. Install web3.")
    WEB3_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TaskData:
    """Task data structure"""
    task_id: int
    user: str
    description: str
    deposit: int
    deadline: int
    status: int
    proof_of_completion: str
    completion_nft_uri: str
    nft_token_id: int
    created_at: int

class SimpleTaskFiAgent:
    """Simplified TaskFi monitoring agent"""
    
    def __init__(self):
        self.setup_models()
        self.setup_blockchain()
        self.setup_integrations()
        
    def setup_models(self):
        """Initialize AI models"""
        logger.info("Setting up AI models...")
        
        if BLIP_AVAILABLE:
            try:
                logger.info("Loading BLIP model...")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                logger.info("BLIP model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load BLIP: {e}")
                self.blip_processor = None
                self.blip_model = None
        else:
            self.blip_processor = None
            self.blip_model = None
            
        if EMBEDDINGS_AVAILABLE:
            try:
                logger.info("Loading sentence transformer...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
            
    def setup_blockchain(self):
        """Initialize blockchain connection"""
        if not WEB3_AVAILABLE:
            logger.error("Web3 not available - blockchain features disabled")
            self.w3 = None
            self.contract = None
            return
            
        logger.info("Setting up blockchain connection...")
        
        try:
            # Ethereum Sepolia configuration
            self.w3 = Web3(Web3.HTTPProvider("https://sepolia.infura.io/v3/f8a842101cf241c68380d2e9a14a2ab3"))
            
            # Test connection
            if self.w3.is_connected():
                logger.info("Connected to Ethereum Sepolia")
                latest_block = self.w3.eth.get_block('latest')
                logger.info(f"Latest block: {latest_block['number']}")
            else:
                logger.error("Failed to connect to Ethereum")
                return
            
            # Contract configuration
            self.contract_address = self.w3.to_checksum_address("0x559B8F2476C923A418114ABFD3704Abf88d43776")
            self.contract_abi = self.load_contract_abi()
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=self.contract_abi
            )
            
            # Set up account for transactions
            self.private_key = os.getenv("PRIVATE_KEY")
            if self.private_key:
                try:
                    self.account = self.w3.eth.account.from_key(self.private_key)
                    logger.info(f"Account configured: {self.account.address}")
                except Exception as e:
                    logger.error(f"Failed to load account: {e}")
                    self.account = None
            else:
                logger.warning("No private key configured - read-only mode")
                self.account = None
            
            logger.info(f"Contract configured at {self.contract_address}")
            
        except Exception as e:
            logger.error(f"Blockchain setup failed: {e}")
            self.w3 = None
            self.contract = None
            
    def setup_integrations(self):
        """Initialize integrations"""
        self.opik_api_key = os.getenv("OPIK_API_KEY", "LkIwLEOFo9heWYYO825wYzRTM")
        self.jenius_token = os.getenv("JENIUS_TOKEN", "")
        logger.info("Integrations configured")
        
    def load_contract_abi(self) -> List[Dict]:
        """Load simplified contract ABI"""
        return [
            {
                "inputs": [{"name": "taskId", "type": "uint256"}],
                "name": "getTask",
                "outputs": [
                    {"name": "taskId", "type": "uint256"},
                    {"name": "user", "type": "address"}, 
                    {"name": "description", "type": "string"},
                    {"name": "deposit", "type": "uint256"},
                    {"name": "deadline", "type": "uint256"},
                    {"name": "status", "type": "uint8"},
                    {"name": "proofOfCompletion", "type": "string"},
                    {"name": "completionNFTUri", "type": "string"},
                    {"name": "nftTokenId", "type": "uint256"},
                    {"name": "createdAt", "type": "uint256"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "getCurrentTaskId", 
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "taskId", "type": "uint256"},
                    {"name": "completionNFTUri", "type": "string"}
                ],
                "name": "approveTaskCompletion",
                "outputs": [],
                "stateMutability": "nonpayable", 
                "type": "function"
            },
            {
                "inputs": [{"name": "taskId", "type": "uint256"}],
                "name": "checkTaskFailure",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
        
    def fetch_tasks(self) -> List[TaskData]:
        """Fetch all tasks from contract"""
        if not self.contract:
            logger.error("Contract not available")
            return []
            
        try:
            logger.info("Fetching tasks from contract...")
            total_tasks = self.contract.functions.getCurrentTaskId().call()
            tasks = []
            
            for task_id in range(total_tasks):
                try:
                    task_data = self.contract.functions.getTask(task_id).call()
                    task = TaskData(
                        task_id=task_data[0],
                        user=task_data[1],
                        description=task_data[2], 
                        deposit=task_data[3],
                        deadline=task_data[4],
                        status=task_data[5],
                        proof_of_completion=task_data[6],
                        completion_nft_uri=task_data[7],
                        nft_token_id=task_data[8],
                        created_at=task_data[9]
                    )
                    tasks.append(task)
                except Exception as e:
                    logger.warning(f"Failed to fetch task {task_id}: {e}")
                    
            logger.info(f"Fetched {len(tasks)} tasks")
            return tasks
            
        except Exception as e:
            logger.error(f"Error fetching tasks: {e}")
            return []
            
    def analyze_image(self, image_url: str) -> str:
        """Analyze image using BLIP"""
        if not self.blip_processor or not self.blip_model:
            return "Image analysis not available - BLIP not loaded"
            
        try:
            logger.info(f"Analyzing image: {image_url}")
            
            # Download image
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            image = Image.open(requests.get(image_url, stream=True).raw)
            
            # Generate caption using BLIP
            inputs = self.blip_processor(image, return_tensors="pt")
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=100)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            logger.info(f"Image caption: {caption}")
            return caption
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return f"Image analysis failed: {e}"
            
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using embeddings"""
        if not self.embedding_model:
            # Fallback: simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0.0
            
        try:
            embedding1 = self.embedding_model.encode(text1)
            embedding2 = self.embedding_model.encode(text2)
            
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
            
    def query_local_llm(self, prompt: str) -> str:
        """Query local Ollama LLM"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"LLM request failed: {response.status_code}")
                return "Error querying LLM"
                
        except Exception as e:
            logger.error(f"Error querying local LLM: {e}")
            # Fallback simple logic
            if "complete" in prompt.lower() or "finish" in prompt.lower():
                return "COMPLETE - Simple keyword match"
            return "INCOMPLETE - Simple keyword check failed"
            
    def verify_task_completion(self, task: TaskData, image_description: str) -> str:
        """Verify if image shows task completion"""
        try:
            # Calculate similarity
            similarity = self.calculate_similarity(task.description, image_description)
            
            # Create verification prompt
            verification_prompt = f"""
            Task Description: {task.description}
            Image Analysis: {image_description}
            Similarity Score: {similarity:.3f}
            
            Based on the task description and image analysis, does this image show evidence of task completion?
            Consider if the image content relates to the task requirements.
            Respond with either "COMPLETE" or "INCOMPLETE" followed by a brief explanation.
            """
            
            llm_response = self.query_local_llm(verification_prompt)
            
            # Determine verification result
            if "COMPLETE" in llm_response.upper() and similarity > 0.3:
                result = "COMPLETE"
            else:
                result = "INCOMPLETE"
                
            logger.info(f"Verification result: {result} (similarity: {similarity:.3f})")
            logger.info(f"LLM response: {llm_response[:100]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in verification: {e}")
            return "INCOMPLETE"
            
    def send_transaction(self, function, description: str):
        """Send a transaction to the contract"""
        if not self.account:
            logger.warning(f"No account configured, skipping: {description}")
            return None
            
        try:
            # Get gas estimate
            gas_estimate = function.estimate_gas({'from': self.account.address})
            
            # Get current gas price
            gas_price = self.w3.eth.gas_price
            
            # Build transaction
            transaction = function.build_transaction({
                'from': self.account.address,
                'gas': gas_estimate,
                'gasPrice': gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            logger.info(f"{description} - Transaction sent: {tx_hash.hex()}")
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            logger.info(f"{description} - Transaction confirmed in block: {receipt.blockNumber}")
            
            return receipt
            
        except Exception as e:
            logger.error(f"Error sending transaction for {description}: {e}")
            return None
            
    def approve_task_completion(self, task_id: int):
        """Approve task completion"""
        if not self.contract:
            return
            
        nft_uri = f"https://ipfs.io/ipfs/QmTaskFi{task_id}Completion{int(time.time())}"
        function = self.contract.functions.approveTaskCompletion(task_id, nft_uri)
        
        return self.send_transaction(function, f"Approve task {task_id}")
        
    def mark_task_failed(self, task_id: int):
        """Mark task as failed"""
        if not self.contract:
            return
            
        function = self.contract.functions.checkTaskFailure(task_id)
        
        return self.send_transaction(function, f"Mark task {task_id} as failed")
        
    def process_tasks_with_proofs(self, tasks: List[TaskData]):
        """Process tasks that have proof submissions"""
        proof_tasks = [
            task for task in tasks 
            if task.status == 0 and task.proof_of_completion  # 0 = InProgress
        ]
        
        logger.info(f"Found {len(proof_tasks)} tasks with proofs to process")
        
        for task in proof_tasks:
            try:
                logger.info(f"Processing task {task.task_id}: {task.description[:50]}...")
                
                # Analyze the proof image
                image_description = self.analyze_image(task.proof_of_completion)
                
                # Verify if it shows completion
                verification_result = self.verify_task_completion(task, image_description)
                
                # Take action based on verification
                if verification_result == "COMPLETE":
                    logger.info(f"Task {task.task_id} verified as complete - approving")
                    self.approve_task_completion(task.task_id)
                else:
                    logger.info(f"Task {task.task_id} verification failed - marking as failed")
                    self.mark_task_failed(task.task_id)
                    
            except Exception as e:
                logger.error(f"Error processing task {task.task_id}: {e}")
                
    def check_expired_tasks(self, tasks: List[TaskData]):
        """Check for expired tasks without proofs"""
        current_time = int(time.time())
        
        expired_tasks = [
            task for task in tasks 
            if (task.status == 0 and  # InProgress
                current_time > task.deadline and 
                not task.proof_of_completion)
        ]
        
        logger.info(f"Found {len(expired_tasks)} expired tasks to mark as failed")
        
        for task in expired_tasks:
            try:
                logger.info(f"Marking expired task {task.task_id} as failed")
                self.mark_task_failed(task.task_id)
            except Exception as e:
                logger.error(f"Error marking task {task.task_id} as failed: {e}")
                
    def log_to_opik(self, data: Dict):
        """Log data to Opik (simplified)"""
        try:
            logger.info(f"Opik log: {json.dumps(data, indent=2)}")
            # Implement actual Opik logging here
        except Exception as e:
            logger.error(f"Error logging to Opik: {e}")
            
    async def run_monitoring_cycle(self):
        """Run one monitoring cycle"""
        try:
            logger.info("=" * 50)
            logger.info("Starting monitoring cycle...")
            
            # Fetch all tasks
            tasks = self.fetch_tasks()
            
            if not tasks:
                logger.info("No tasks found")
                return
                
            # Process tasks with proofs
            self.process_tasks_with_proofs(tasks)
            
            # Check for expired tasks
            self.check_expired_tasks(tasks)
            
            # Log results
            self.log_to_opik({
                "timestamp": datetime.now().isoformat(),
                "total_tasks": len(tasks),
                "tasks_with_proofs": len([t for t in tasks if t.proof_of_completion and t.status == 0]),
                "expired_tasks": len([t for t in tasks if t.status == 0 and int(time.time()) > t.deadline and not t.proof_of_completion])
            })
            
            logger.info("Monitoring cycle completed")
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
            
    async def run_forever(self, interval: int = 20):
        """Run monitoring agent forever"""
        logger.info(f"🤖 Starting TaskFi monitoring agent")
        logger.info(f"📊 Checking every {interval} seconds")
        logger.info(f"📍 Contract: {self.contract_address if self.contract else 'Not available'}")
        logger.info(f"🎯 Features: BLIP={BLIP_AVAILABLE}, Embeddings={EMBEDDINGS_AVAILABLE}, Web3={WEB3_AVAILABLE}")
        logger.info("=" * 50)
        
        while True:
            try:
                await self.run_monitoring_cycle()
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Agent stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await asyncio.sleep(interval)

async def main():
    """Main function"""
    # Check environment
    missing_vars = []
    if not os.getenv("PRIVATE_KEY"):
        missing_vars.append("PRIVATE_KEY")
        
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.warning("Agent will run in read-only mode")
        
    # Create and run agent
    agent = SimpleTaskFiAgent()
    await agent.run_forever(interval=20)

if __name__ == "__main__":
    asyncio.run(main())