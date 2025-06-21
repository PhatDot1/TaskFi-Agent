#!/usr/bin/env python3
"""
GitHub TaskFi Monitoring Agent
Handles tasks with GitHub repository proof links using LlamaIndex
"""

import os
import asyncio
import time
import json
import requests
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Core dependencies
try:
    from web3 import Web3
    WEB3_AVAILABLE = True
except ImportError:
    print("Warning: Web3 not available. Install web3.")
    WEB3_AVAILABLE = False

# LlamaIndex dependencies
try:
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.readers.github import GithubRepositoryReader, GithubClient
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    import nest_asyncio
    nest_asyncio.apply()
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    print("Warning: LlamaIndex not available. Install llama-index and llama-index-readers-github.")
    LLAMAINDEX_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TaskData:
    """Task data structure matching contract output"""
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

class GitHubTaskFiAgent:
    """GitHub-specific TaskFi monitoring agent using LlamaIndex"""
    
    def __init__(self):
        self.setup_llamaindex()
        self.setup_blockchain()
        self.setup_integrations()
        
    def setup_llamaindex(self):
        """Initialize LlamaIndex with OpenAI"""
        if not LLAMAINDEX_AVAILABLE:
            logger.error("LlamaIndex not available - GitHub analysis disabled")
            self.github_client = None
            return
            
        logger.info("Setting up LlamaIndex for GitHub analysis...")
        
        # Setup OpenAI
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY not found - GitHub analysis disabled")
            self.github_client = None
            return
            
        # Configure LlamaIndex settings
        Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
        Settings.embed_model = OpenAIEmbedding(api_key=openai_api_key)
        
        # Setup GitHub client
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            self.github_client = GithubClient(github_token=github_token, verbose=False)
            logger.info("GitHub client configured with token")
        else:
            logger.warning("No GITHUB_TOKEN found - using public access only")
            self.github_client = GithubClient(verbose=False)
            
        logger.info("LlamaIndex setup completed")
        
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
            self.contract_address = self.w3.to_checksum_address("0xBB28f99330B5fDffd96a1D1D5D6f94345B6e1229")
            self.contract_abi = self.load_contract_abi()
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=self.contract_abi
            )
            
            # Set up account for transactions
            self.private_key = os.getenv("PRIVATE_KEY")
            if self.private_key:
                try:
                    # Remove 0x prefix if present
                    if self.private_key.startswith('0x'):
                        self.private_key = self.private_key[2:]
                    
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
        """Load actual contract ABI from file"""
        try:
            # Try to load from abi/TaskFi.json
            abi_path = Path("abi/TaskFi.json")
            if abi_path.exists():
                with open(abi_path, 'r') as f:
                    abi_data = json.load(f)
                    if 'abi' in abi_data:
                        logger.info("Loaded ABI from abi/TaskFi.json")
                        return abi_data['abi']
                    else:
                        logger.info("Using ABI array from abi/TaskFi.json")
                        return abi_data
            else:
                logger.warning("ABI file not found, using simplified ABI")
                return self.get_simplified_abi()
        except Exception as e:
            logger.error(f"Failed to load ABI: {e}")
            return self.get_simplified_abi()
            
    def get_simplified_abi(self) -> List[Dict]:
        """Simplified ABI for basic functionality"""
        return [
            {
                "inputs": [],
                "name": "getCurrentTaskId",
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"name": "taskId", "type": "uint256"}],
                "name": "getTask",
                "outputs": [{
                    "components": [
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
                    "internalType": "struct TaskFi.Task",
                    "name": "",
                    "type": "tuple"
                }],
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
        
    def is_github_link(self, url: str) -> bool:
        """Check if URL is a GitHub repository link"""
        if not url:
            return False
            
        github_patterns = [
            r"https://github\.com/[\w\-\.]+/[\w\-\.]+",
            r"https://www\.github\.com/[\w\-\.]+/[\w\-\.]+",
        ]
        
        return any(re.match(pattern, url) for pattern in github_patterns)
        
    def parse_github_url(self, url: str) -> tuple:
        """Parse GitHub URL to extract owner and repo"""
        try:
            # Remove trailing slash and .git
            clean_url = url.rstrip('/').replace('.git', '')
            
            # Parse URL
            parsed = urlparse(clean_url)
            path_parts = parsed.path.strip('/').split('/')
            
            if len(path_parts) >= 2:
                owner = path_parts[0]
                repo = path_parts[1]
                return owner, repo
            else:
                raise ValueError("Invalid GitHub URL format")
                
        except Exception as e:
            logger.error(f"Error parsing GitHub URL {url}: {e}")
            return None, None
            
    def analyze_github_repo(self, github_url: str, task_description: str) -> str:
        """Analyze GitHub repository using LlamaIndex"""
        if not self.github_client:
            return "GitHub analysis not available - LlamaIndex not configured"
            
        try:
            logger.info(f"Analyzing GitHub repo: {github_url}")
            
            # Parse GitHub URL
            owner, repo = self.parse_github_url(github_url)
            if not owner or not repo:
                return "Invalid GitHub URL format"
                
            logger.info(f"Analyzing repo: {owner}/{repo}")
            
            # Load repository documents
            documents = GithubRepositoryReader(
                github_client=self.github_client,
                owner=owner,
                repo=repo,
                use_parser=False,
                verbose=False,
                filter_file_extensions=(
                    [
                        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
                        ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib",
                        ".zip", ".tar", ".gz", ".rar", ".7z",
                        ".mp4", ".avi", ".mov", ".wmv", ".flv",
                        ".mp3", ".wav", ".ogg", ".flac"
                    ],
                    GithubRepositoryReader.FilterType.EXCLUDE,
                ),
            ).load_data(branch="main")
            
            if not documents:
                # Try master branch if main doesn't exist
                documents = GithubRepositoryReader(
                    github_client=self.github_client,
                    owner=owner,
                    repo=repo,
                    use_parser=False,
                    verbose=False,
                    filter_file_extensions=(
                        [
                            ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
                            ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib",
                            ".zip", ".tar", ".gz", ".rar", ".7z",
                            ".mp4", ".avi", ".mov", ".wmv", ".flv",
                            ".mp3", ".wav", ".ogg", ".flac"
                        ],
                        GithubRepositoryReader.FilterType.EXCLUDE,
                    ),
                ).load_data(branch="master")
            
            if not documents:
                return f"No accessible documents found in repository {owner}/{repo}"
                
            logger.info(f"Loaded {len(documents)} documents from repository")
            
            # Create vector index
            index = VectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine()
            
            # Create analysis query
            analysis_query = f"""
            Task Description: {task_description}
            
            Based on the code, documentation, and files in this repository, does this repository demonstrate completion of the task described above?
            
            Please analyze:
            1. What does this repository contain?
            2. How does it relate to the task requirements?
            3. Does it show evidence of task completion?
            4. What specific features or implementations are relevant?
            
            Respond with "COMPLETE" if the repository demonstrates task completion, or "INCOMPLETE" if it doesn't.
            Include a brief explanation of your reasoning.
            """
            
            # Query the repository
            response = query_engine.query(analysis_query)
            
            logger.info(f"GitHub analysis complete: {str(response)[:200]}...")
            return str(response)
            
        except Exception as e:
            logger.error(f"Error analyzing GitHub repository: {e}")
            return f"GitHub analysis failed: {e}"
            
    def verify_github_completion(self, task: TaskData) -> str:
        """Verify if GitHub repository shows task completion"""
        try:
            # Analyze the GitHub repository
            github_analysis = self.analyze_github_repo(task.proof_of_completion, task.description)
            
            # Determine verification result based on analysis
            if "COMPLETE" in github_analysis.upper():
                result = "COMPLETE"
            else:
                result = "INCOMPLETE"
                
            logger.info(f"GitHub verification result: {result}")
            logger.info(f"Analysis summary: {github_analysis[:300]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in GitHub verification: {e}")
            return "INCOMPLETE"
            
    def fetch_tasks(self) -> List[TaskData]:
        """Fetch all tasks from contract"""
        if not self.contract:
            logger.error("Contract not available")
            return []
            
        try:
            logger.info("Fetching tasks from contract...")
            total_tasks = self.contract.functions.getCurrentTaskId().call()
            logger.info(f"Total tasks in contract: {total_tasks}")
            tasks = []
            
            for task_id in range(total_tasks):
                try:
                    # Call getTask which returns a tuple
                    task_tuple = self.contract.functions.getTask(task_id).call()
                    
                    # Parse the tuple into TaskData
                    task = TaskData(
                        task_id=task_tuple[0],
                        user=task_tuple[1],
                        description=task_tuple[2],
                        deposit=task_tuple[3],
                        deadline=task_tuple[4],
                        status=task_tuple[5],
                        proof_of_completion=task_tuple[6],
                        completion_nft_uri=task_tuple[7],
                        nft_token_id=task_tuple[8],
                        created_at=task_tuple[9]
                    )
                    tasks.append(task)
                    logger.info(f"✅ Task {task_id}: {task.description[:50]}... Status: {task.status}")
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch task {task_id}: {e}")
                    
            logger.info(f"Successfully fetched {len(tasks)} tasks")
            return tasks
            
        except Exception as e:
            logger.error(f"Error fetching tasks: {e}")
            return []
            
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
            
        nft_uri = f"https://ipfs.io/ipfs/QmTaskFiGitHub{task_id}Completion{int(time.time())}"
        function = self.contract.functions.approveTaskCompletion(task_id, nft_uri)
        
        return self.send_transaction(function, f"Approve GitHub task {task_id}")
        
    def mark_task_failed(self, task_id: int):
        """Mark task as failed"""
        if not self.contract:
            return
            
        function = self.contract.functions.checkTaskFailure(task_id)
        
        return self.send_transaction(function, f"Mark GitHub task {task_id} as failed")
        
    def process_github_tasks(self, tasks: List[TaskData]):
        """Process tasks that have GitHub proof submissions"""
        github_tasks = [
            task for task in tasks 
            if (task.status == 0 and task.proof_of_completion and 
                self.is_github_link(task.proof_of_completion))  # 0 = InProgress
        ]
        
        logger.info(f"Found {len(github_tasks)} tasks with GitHub proofs to process")
        
        for task in github_tasks:
            try:
                logger.info(f"🐙 Processing GitHub task {task.task_id}: {task.description[:50]}...")
                logger.info(f"📂 GitHub URL: {task.proof_of_completion}")
                
                # Verify if GitHub repo shows completion
                verification_result = self.verify_github_completion(task)
                
                # Take action based on verification
                if verification_result == "COMPLETE":
                    logger.info(f"✅ GitHub task {task.task_id} verified as complete - approving")
                    self.approve_task_completion(task.task_id)
                else:
                    logger.info(f"❌ GitHub task {task.task_id} verification failed - marking as failed")
                    self.mark_task_failed(task.task_id)
                    
            except Exception as e:
                logger.error(f"Error processing GitHub task {task.task_id}: {e}")
                
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
                logger.info(f"⏰ Marking expired task {task.task_id} as failed")
                self.mark_task_failed(task.task_id)
            except Exception as e:
                logger.error(f"Error marking task {task.task_id} as failed: {e}")
                
    async def run_monitoring_cycle(self):
        """Run one monitoring cycle"""
        try:
            logger.info("=" * 60)
            logger.info("🐙 Starting GitHub monitoring cycle...")
            
            # Fetch all tasks
            tasks = self.fetch_tasks()
            
            if not tasks:
                logger.info("📭 No tasks found")
                return
                
            # Show task summary
            status_counts = {}
            github_proof_count = 0
            
            for task in tasks:
                status_name = ["InProgress", "Complete", "Failed"][task.status]
                status_counts[status_name] = status_counts.get(status_name, 0) + 1
                
                if task.proof_of_completion and self.is_github_link(task.proof_of_completion):
                    github_proof_count += 1
                    
            logger.info(f"📊 Task Summary: {status_counts}")
            logger.info(f"🐙 GitHub proofs found: {github_proof_count}")
                
            # Process tasks with GitHub proofs
            self.process_github_tasks(tasks)
            
            # Check for expired tasks
            self.check_expired_tasks(tasks)
            
            logger.info("✅ GitHub monitoring cycle completed")
            
        except Exception as e:
            logger.error(f"❌ Error in GitHub monitoring cycle: {e}")
            
    async def run_forever(self, interval: int = 30):
        """Run monitoring agent forever"""
        logger.info("🐙 Starting GitHub TaskFi monitoring agent")
        logger.info(f"📊 Checking every {interval} seconds")
        logger.info(f"📍 Contract: {self.contract_address if self.contract else 'Not available'}")
        logger.info(f"🎯 Features: LlamaIndex={LLAMAINDEX_AVAILABLE}, Web3={WEB3_AVAILABLE}")
        logger.info(f"🔑 Account: {self.account.address if self.account else 'Read-only mode'}")
        logger.info(f"🐙 GitHub Token: {'✅ Configured' if os.getenv('GITHUB_TOKEN') else '❌ Public access only'}")
        logger.info("=" * 60)
        
        while True:
            try:
                await self.run_monitoring_cycle()
                logger.info(f"😴 Waiting {interval} seconds until next check...")
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 GitHub agent stopped by user")
                break
            except Exception as e:
                logger.error(f"💥 Unexpected error: {e}")
                await asyncio.sleep(interval)

async def main():
    """Main function"""
    # Check environment
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        logger.error("🔧 GitHub agent requires OPENAI_API_KEY for LlamaIndex")
        return
        
    optional_vars = ["PRIVATE_KEY", "GITHUB_TOKEN"]
    missing_optional = [var for var in optional_vars if not os.getenv(var)]
    
    if missing_optional:
        logger.warning(f"⚠️ Missing optional environment variables: {missing_optional}")
        if "PRIVATE_KEY" in missing_optional:
            logger.warning("🔍 Agent will run in read-only mode")
        if "GITHUB_TOKEN" in missing_optional:
            logger.warning("🐙 Agent will use GitHub public access only")
        
    # Create and run agent
    agent = GitHubTaskFiAgent()
    await agent.run_forever(interval=30)

if __name__ == "__main__":
    asyncio.run(main())