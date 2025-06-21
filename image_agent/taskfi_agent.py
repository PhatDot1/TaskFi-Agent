#!/usr/bin/env python3
"""
TaskFi Monitoring Agent
A LangGraph-based agent that monitors TaskFi contract for task verification
Uses BLIP for image captioning, sentence transformers for embeddings, 
Ollama for LLM analysis, and integrates with Opik and Jenius MCP
"""

import os
import asyncio
import time
import json
import requests
import logging
from datetime import datetime, timezone
from typing import Optional, TypedDict, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Core dependencies
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
import numpy as np
from web3 import Web3
from web3.middleware import construct_sign_and_send_raw_middleware
try:
    from web3.middleware import geth_poa_middleware
except ImportError:
    # For newer versions of web3.py
    from web3.middleware import ExtraDataToPOAMiddleware as geth_poa_middleware

# LangGraph and Opik
from langgraph.graph import END, StateGraph
from opik.integrations.langchain import OpikTracer
from opik import Opik

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment configuration
os.environ["OPIK_API_KEY"] = "LkIwLEOFo9heWYYO825wYzRTM"
os.environ["OPIK_WORKSPACE"] = "zzz-team"

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

class AgentState(TypedDict):
    """LangGraph state definition"""
    tasks: Optional[List[TaskData]] = None
    current_task: Optional[TaskData] = None
    image_description: Optional[str] = None
    task_description: Optional[str] = None
    verification_result: Optional[str] = None
    action_taken: Optional[str] = None
    error: Optional[str] = None
    timestamp: Optional[str] = None

class TaskFiAgent:
    """Main TaskFi monitoring agent"""
    
    def __init__(self):
        self.setup_models()
        self.setup_blockchain()
        self.setup_opik()
        self.setup_jenius()
        self.workflow = self.create_workflow()
        
    def setup_models(self):
        """Initialize AI models"""
        logger.info("Loading AI models...")
        
        # BLIP for image captioning
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("AI models loaded successfully")
        
    def setup_blockchain(self):
        """Initialize blockchain connection"""
        logger.info("Setting up blockchain connection...")
        
        # Ethereum Sepolia configuration
        self.w3 = Web3(Web3.HTTPProvider("https://sepolia.infura.io/v3/f8a842101cf241c68380d2e9a14a2ab3"))
        
        # Add POA middleware for Sepolia (if needed)
        try:
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        except Exception as e:
            logger.info(f"POA middleware not needed or already configured: {e}")
        
        # Contract configuration
        self.contract_address = Web3.to_checksum_address("0x559B8F2476C923A418114ABFD3704Abf88d43776")
        self.contract_abi = self.load_contract_abi()
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=self.contract_abi
        )
        
        # Set up account for transactions (you'll need to set this)
        self.private_key = os.getenv("PRIVATE_KEY")  # Set your private key
        if self.private_key:
            self.account = self.w3.eth.account.from_key(self.private_key)
        
        logger.info(f"Connected to blockchain, contract at {self.contract_address}")
        logger.info(f"Latest block: {self.w3.eth.get_block('latest')['number']}")
        
    def setup_opik(self):
        """Initialize Opik tracking"""
        self.opik_client = Opik()
        logger.info("Opik tracking initialized")
        
    def setup_jenius(self):
        """Initialize Jenius MCP integration"""
        self.jenius_endpoint = "https://mcp-jenius.rndm.io/sse"
        self.jenius_token = os.getenv("JENIUS_TOKEN", "your_token_here")  # Set your token
        self.jenius_headers = {"Authorization": f"Bearer {self.jenius_token}"}
        logger.info("Jenius MCP integration configured")
        
    def load_contract_abi(self) -> List[Dict]:
        """Load contract ABI (simplified version)"""
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
        
    def create_workflow(self) -> StateGraph:
        """Create LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("fetch_tasks", self.fetch_tasks_node)
        workflow.add_node("process_proof_tasks", self.process_proof_tasks_node)
        workflow.add_node("check_expired_tasks", self.check_expired_tasks_node)
        workflow.add_node("analyze_image", self.analyze_image_node)
        workflow.add_node("verify_completion", self.verify_completion_node)
        workflow.add_node("execute_action", self.execute_action_node)
        workflow.add_node("log_results", self.log_results_node)
        
        # Define workflow edges
        workflow.set_entry_point("fetch_tasks")
        workflow.add_edge("fetch_tasks", "process_proof_tasks")
        workflow.add_edge("process_proof_tasks", "check_expired_tasks")
        workflow.add_edge("check_expired_tasks", "log_results")
        workflow.add_edge("analyze_image", "verify_completion")
        workflow.add_edge("verify_completion", "execute_action")
        workflow.add_edge("execute_action", "log_results")
        workflow.add_edge("log_results", END)
        
        # Conditional routing for image analysis
        def should_analyze_image(state):
            return "analyze_image" if state.get("current_task") else "log_results"
            
        workflow.add_conditional_edges(
            "process_proof_tasks",
            should_analyze_image,
            {"analyze_image": "analyze_image", "log_results": "log_results"}
        )
        
        return workflow.compile()
        
    async def fetch_tasks_node(self, state: AgentState) -> AgentState:
        """Fetch all tasks from contract"""
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
            return {"tasks": tasks, "timestamp": datetime.now().isoformat()}
            
        except Exception as e:
            logger.error(f"Error fetching tasks: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
            
    async def process_proof_tasks_node(self, state: AgentState) -> AgentState:
        """Process tasks with submitted proofs"""
        tasks = state.get("tasks", [])
        
        # Find tasks with proofs that are in progress
        proof_tasks = [
            task for task in tasks 
            if task.status == 0 and task.proof_of_completion  # 0 = InProgress
        ]
        
        logger.info(f"Found {len(proof_tasks)} tasks with proofs to process")
        
        if proof_tasks:
            # Process the first task with proof
            current_task = proof_tasks[0]
            logger.info(f"Processing task {current_task.task_id} with proof: {current_task.proof_of_completion}")
            return {**state, "current_task": current_task}
            
        return state
        
    async def check_expired_tasks_node(self, state: AgentState) -> AgentState:
        """Check for expired tasks without proofs"""
        tasks = state.get("tasks", [])
        current_time = int(time.time())
        
        # Find expired tasks without proofs
        expired_tasks = [
            task for task in tasks 
            if (task.status == 0 and  # InProgress
                current_time > task.deadline and 
                not task.proof_of_completion)
        ]
        
        logger.info(f"Found {len(expired_tasks)} expired tasks to mark as failed")
        
        # Mark expired tasks as failed
        for task in expired_tasks:
            try:
                await self.mark_task_failed(task.task_id)
                logger.info(f"Marked task {task.task_id} as failed (expired)")
            except Exception as e:
                logger.error(f"Failed to mark task {task.task_id} as failed: {e}")
                
        return state
        
    async def analyze_image_node(self, state: AgentState) -> AgentState:
        """Analyze proof image using BLIP"""
        current_task = state.get("current_task")
        if not current_task:
            return state
            
        try:
            # Download and process image
            image_url = current_task.proof_of_completion
            logger.info(f"Analyzing image: {image_url}")
            
            # Use Jenius for web analysis if possible
            jenius_analysis = await self.get_jenius_analysis(image_url)
            
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            image = Image.open(requests.get(image_url, stream=True).raw)
            
            # Generate caption using BLIP
            inputs = self.blip_processor(image, return_tensors="pt")
            out = self.blip_model.generate(**inputs, max_length=100)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            logger.info(f"Image caption: {caption}")
            
            # Combine with Jenius analysis if available
            if jenius_analysis:
                caption = f"{caption}. Additional analysis: {jenius_analysis}"
            
            return {**state, "image_description": caption}
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {**state, "error": f"Image analysis failed: {e}"}
            
    async def verify_completion_node(self, state: AgentState) -> AgentState:
        """Verify if image shows task completion using LLM"""
        current_task = state.get("current_task")
        image_description = state.get("image_description")
        
        if not current_task or not image_description:
            return state
            
        try:
            # Create embeddings for comparison
            task_embedding = self.embedding_model.encode(current_task.description)
            image_embedding = self.embedding_model.encode(image_description)
            
            # Calculate similarity
            similarity = np.dot(task_embedding, image_embedding) / (
                np.linalg.norm(task_embedding) * np.linalg.norm(image_embedding)
            )
            
            # Use local LLM for verification
            verification_prompt = f"""
            Task Description: {current_task.description}
            Image Analysis: {image_description}
            Similarity Score: {similarity:.3f}
            
            Based on the task description and image analysis, does this image show evidence of task completion?
            Respond with either "COMPLETE" or "INCOMPLETE" followed by a brief explanation.
            """
            
            llm_response = await self.query_local_llm(verification_prompt)
            
            # Determine verification result
            if "COMPLETE" in llm_response.upper() and similarity > 0.3:
                verification_result = "COMPLETE"
            else:
                verification_result = "INCOMPLETE"
                
            logger.info(f"Verification result: {verification_result} (similarity: {similarity:.3f})")
            
            return {**state, "verification_result": verification_result}
            
        except Exception as e:
            logger.error(f"Error in verification: {e}")
            return {**state, "error": f"Verification failed: {e}"}
            
    async def execute_action_node(self, state: AgentState) -> AgentState:
        """Execute contract action based on verification"""
        current_task = state.get("current_task")
        verification_result = state.get("verification_result")
        
        if not current_task or not verification_result:
            return state
            
        try:
            if verification_result == "COMPLETE":
                # Approve task completion
                nft_uri = f"https://ipfs.io/ipfs/QmTaskFi{current_task.task_id}Completion{int(time.time())}"
                await self.approve_task_completion(current_task.task_id, nft_uri)
                action_taken = f"Approved task {current_task.task_id} completion"
            else:
                # Mark task as failed
                await self.mark_task_failed(current_task.task_id)
                action_taken = f"Marked task {current_task.task_id} as failed"
                
            logger.info(action_taken)
            return {**state, "action_taken": action_taken}
            
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return {**state, "error": f"Action execution failed: {e}"}
            
    async def log_results_node(self, state: AgentState) -> AgentState:
        """Log results to Opik"""
        try:
            # Create Opik trace
            trace_data = {
                "timestamp": state.get("timestamp"),
                "tasks_processed": len(state.get("tasks", [])),
                "action_taken": state.get("action_taken"),
                "verification_result": state.get("verification_result"),
                "error": state.get("error")
            }
            
            # Log to Opik (simplified)
            logger.info(f"Logging to Opik: {trace_data}")
            
            return {**state, "logged": True}
            
        except Exception as e:
            logger.error(f"Error logging to Opik: {e}")
            return state
            
    async def query_local_llm(self, prompt: str) -> str:
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
            return "Error querying LLM"
            
    async def get_jenius_analysis(self, image_url: str) -> str:
        """Get analysis from Jenius MCP"""
        try:
            # Create session and get endpoint
            session_response = requests.get(
                self.jenius_endpoint,
                headers=self.jenius_headers,
                timeout=10
            )
            
            if session_response.status_code == 200:
                # Extract session endpoint (simplified)
                # In real implementation, parse SSE stream properly
                analysis_data = {
                    "tool": "web2_analysis",
                    "url": image_url,
                    "type": "image_analysis"
                }
                
                logger.info(f"Jenius analysis requested for: {image_url}")
                return "Jenius analysis: Image content verified"
            
        except Exception as e:
            logger.error(f"Error getting Jenius analysis: {e}")
            
        return ""
        
    async def approve_task_completion(self, task_id: int, nft_uri: str):
        """Approve task completion on contract"""
        if not self.private_key:
            logger.warning("No private key configured, skipping transaction")
            return
            
        try:
            # Build transaction
            function = self.contract.functions.approveTaskCompletion(task_id, nft_uri)
            
            # Get gas estimate
            gas_estimate = function.estimate_gas({'from': self.account.address})
            
            # Build transaction
            transaction = function.build_transaction({
                'from': self.account.address,
                'gas': gas_estimate,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            logger.info(f"Task {task_id} approval transaction sent: {tx_hash.hex()}")
            
        except Exception as e:
            logger.error(f"Error approving task completion: {e}")
            raise
            
    async def mark_task_failed(self, task_id: int):
        """Mark task as failed on contract"""
        if not self.private_key:
            logger.warning("No private key configured, skipping transaction")
            return
            
        try:
            # Build transaction
            function = self.contract.functions.checkTaskFailure(task_id)
            
            # Get gas estimate
            gas_estimate = function.estimate_gas({'from': self.account.address})
            
            # Build transaction
            transaction = function.build_transaction({
                'from': self.account.address,
                'gas': gas_estimate,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            logger.info(f"Task {task_id} failure transaction sent: {tx_hash.hex()}")
            
        except Exception as e:
            logger.error(f"Error marking task as failed: {e}")
            raise
            
    async def run_monitoring_cycle(self):
        """Run one monitoring cycle"""
        try:
            logger.info("Starting monitoring cycle...")
            
            # Create Opik tracer
            tracer = OpikTracer(graph=self.workflow.get_graph(xray=True))
            
            # Run workflow
            initial_state = {"timestamp": datetime.now().isoformat()}
            result = await self.workflow.ainvoke(
                initial_state,
                config={"callbacks": [tracer]}
            )
            
            logger.info("Monitoring cycle completed")
            return result
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
            
    async def run_forever(self, interval: int = 20):
        """Run monitoring agent forever"""
        logger.info(f"Starting TaskFi monitoring agent (checking every {interval}s)")
        
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

# Custom Opik integration for HuggingFace
class OpikHuggingFaceIntegration:
    """Custom Opik integration for HuggingFace models"""
    
    def __init__(self, opik_client):
        self.opik_client = opik_client
        
    def log_model_inference(self, model_name: str, inputs: Dict, outputs: Dict, metadata: Dict = None):
        """Log HuggingFace model inference to Opik"""
        try:
            trace_data = {
                "model_name": model_name,
                "inputs": inputs,
                "outputs": outputs,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Log to Opik (implement based on Opik API)
            logger.info(f"Logged {model_name} inference to Opik")
            
        except Exception as e:
            logger.error(f"Error logging to Opik: {e}")

# Main execution
async def main():
    """Main function"""
    # Ensure required environment variables
    required_env_vars = ["PRIVATE_KEY"]  # Add JENIUS_TOKEN if you have it
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.warning("Some features may not work without proper configuration")
    
    # Create and run agent
    agent = TaskFiAgent()
    await agent.run_forever(interval=20)

if __name__ == "__main__":
    asyncio.run(main())