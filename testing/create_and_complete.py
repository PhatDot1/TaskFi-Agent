#!/usr/bin/env python3
"""
Create Test Task for TaskFi Agent
Creates a task "create an art of an enlightened male being" 
and submits proof image for testing the monitoring agent
"""

import os
import asyncio
import time
from dotenv import load_dotenv
from web3 import Web3
import json
from pathlib import Path

load_dotenv()

class TaskFiTester:
    def __init__(self):
        self.setup_web3()
        self.setup_contract()
        
    def setup_web3(self):
        """Setup Web3 connection"""
        self.w3 = Web3(Web3.HTTPProvider("https://sepolia.infura.io/v3/f8a842101cf241c68380d2e9a14a2ab3"))
        
        if not self.w3.is_connected():
            raise Exception("Failed to connect to Ethereum Sepolia")
            
        print(f"✅ Connected to Ethereum Sepolia")
        print(f"📦 Latest block: {self.w3.eth.get_block('latest')['number']}")
        
    def setup_contract(self):
        """Setup contract connection"""
        # Load ABI
        abi_path = Path("abi/TaskFi.json")
        if abi_path.exists():
            with open(abi_path, 'r') as f:
                abi_data = json.load(f)
                self.abi = abi_data['abi'] if 'abi' in abi_data else abi_data
        else:
            raise Exception("ABI file not found at abi/TaskFi.json")
            
        # Setup contract
        self.contract_address = self.w3.to_checksum_address("0x559B8F2476C923A418114ABFD3704Abf88d43776")
        self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.abi)
        
        # Setup account
        private_key = os.getenv("PRIVATE_KEY")
        if not private_key:
            raise Exception("PRIVATE_KEY not found in environment variables")
            
        if private_key.startswith('0x'):
            private_key = private_key[2:]
            
        self.account = self.w3.eth.account.from_key(private_key)
        
        # Check balance
        balance = self.w3.eth.get_balance(self.account.address)
        print(f"🔑 Account: {self.account.address}")
        print(f"💰 Balance: {self.w3.from_wei(balance, 'ether')} ETH")
        
        if balance < self.w3.to_wei(0.02, 'ether'):
            print("⚠️ Warning: Low balance, might not be enough for task deposit + gas")
            
    def send_transaction(self, function, description: str, value: int = 0):
        """Send a transaction to the contract"""
        try:
            print(f"🚀 {description}...")
            
            # Get gas estimate
            gas_estimate = function.estimate_gas({
                'from': self.account.address,
                'value': value
            })
            
            # Get current gas price
            gas_price = self.w3.eth.gas_price
            
            # Build transaction
            transaction = function.build_transaction({
                'from': self.account.address,
                'gas': gas_estimate,
                'gasPrice': gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'value': value
            })
            
            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.raw_transaction)
            
            print(f"📝 Transaction sent: {tx_hash.hex()}")
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            print(f"✅ Transaction confirmed in block: {receipt.blockNumber}")
            
            return receipt
            
        except Exception as e:
            print(f"❌ Error sending transaction: {e}")
            return None
            
    def submit_task(self):
        """Submit the test task"""
        print("\n" + "="*60)
        print("📋 STEP 1: Submitting Test Task")
        print("="*60)
        
        description = "Create an art of an enlightened male being"
        timeline_hours = 24  # 24 hours
        deposit = self.w3.to_wei(0.011, 'ether')  # 0.011 ETH (above minimum)
        
        print(f"📝 Task Description: {description}")
        print(f"⏰ Timeline: {timeline_hours} hours")
        print(f"💰 Deposit: {self.w3.from_wei(deposit, 'ether')} ETH")
        
        # Get current task count before submission
        current_tasks = self.contract.functions.getCurrentTaskId().call()
        print(f"📊 Current tasks before submission: {current_tasks}")
        
        # Submit task
        function = self.contract.functions.submitTask(description, timeline_hours)
        receipt = self.send_transaction(function, "Submitting task", value=deposit)
        
        if receipt:
            # Get new task ID
            new_task_id = current_tasks  # The next task ID will be current count
            
            # Verify task was created
            try:
                task_tuple = self.contract.functions.getTask(new_task_id).call()
                print(f"\n✅ Task created successfully!")
                print(f"🆔 Task ID: {new_task_id}")
                print(f"👤 Owner: {task_tuple[1]}")
                print(f"📝 Description: {task_tuple[2]}")
                print(f"💰 Deposit: {self.w3.from_wei(task_tuple[3], 'ether')} ETH")
                print(f"📊 Status: {['InProgress', 'Complete', 'Failed'][task_tuple[5]]}")
                
                return new_task_id
                
            except Exception as e:
                print(f"❌ Error verifying task creation: {e}")
                return None
        else:
            return None
            
    def submit_proof(self, task_id: int):
        """Submit proof of completion"""
        print("\n" + "="*60)
        print("📸 STEP 2: Submitting Proof of Completion")
        print("="*60)
        
        proof_url = "https://gateway.pinata.cloud/ipfs/QmQRAK6oDejNqCGEQEPwtSQRHw3eL8bkUf93odRnuFAmji"
        
        print(f"🎯 Task ID: {task_id}")
        print(f"🖼️ Proof Image: {proof_url}")
        
        # Check task status first
        try:
            task_tuple = self.contract.functions.getTask(task_id).call()
            if task_tuple[5] != 0:  # 0 = InProgress
                print(f"❌ Task is not in progress (status: {['InProgress', 'Complete', 'Failed'][task_tuple[5]]})")
                return False
                
            if task_tuple[1].lower() != self.account.address.lower():
                print(f"❌ You are not the owner of this task!")
                print(f"   Task owner: {task_tuple[1]}")
                print(f"   Your address: {self.account.address}")
                return False
                
        except Exception as e:
            print(f"❌ Error checking task: {e}")
            return False
            
        # Submit proof
        function = self.contract.functions.submitProof(task_id, proof_url)
        receipt = self.send_transaction(function, "Submitting proof of completion")
        
        if receipt:
            # Verify proof was submitted
            try:
                task_tuple = self.contract.functions.getTask(task_id).call()
                print(f"\n✅ Proof submitted successfully!")
                print(f"🔗 Proof URL: {task_tuple[6]}")
                
                return True
                
            except Exception as e:
                print(f"❌ Error verifying proof submission: {e}")
                return False
        else:
            return False
            
    def check_task_status(self, task_id: int):
        """Check the current status of the task"""
        print("\n" + "="*60)
        print("📊 STEP 3: Final Task Status Check")
        print("="*60)
        
        try:
            task_tuple = self.contract.functions.getTask(task_id).call()
            
            print(f"🆔 Task ID: {task_id}")
            print(f"👤 Owner: {task_tuple[1]}")
            print(f"📝 Description: {task_tuple[2]}")
            print(f"💰 Deposit: {self.w3.from_wei(task_tuple[3], 'ether')} ETH")
            print(f"⏰ Deadline: {task_tuple[4]} (timestamp)")
            print(f"📊 Status: {['InProgress', 'Complete', 'Failed'][task_tuple[5]]}")
            print(f"🔗 Proof: {task_tuple[6] if task_tuple[6] else 'None'}")
            print(f"🎨 NFT URI: {task_tuple[7] if task_tuple[7] else 'None'}")
            print(f"🎯 NFT Token ID: {task_tuple[8] if task_tuple[8] > 0 else 'None'}")
            
            return task_tuple
            
        except Exception as e:
            print(f"❌ Error checking task status: {e}")
            return None

async def main():
    """Main function"""
    print("🎨 TaskFi Test Task Creator")
    print("Creating task: 'Create an art of an enlightened male being'")
    print("Will submit proof image for agent testing")
    print("="*60)
    
    try:
        tester = TaskFiTester()
        
        # Step 1: Submit task
        task_id = tester.submit_task()
        if task_id is None:
            print("❌ Failed to submit task")
            return
            
        # Step 2: Submit proof
        success = tester.submit_proof(task_id)
        if not success:
            print("❌ Failed to submit proof")
            return
            
        # Step 3: Check final status
        tester.check_task_status(task_id)
        
        print("\n" + "="*60)
        print("🎉 TEST TASK SETUP COMPLETE!")
        print("="*60)
        print(f"✅ Task ID {task_id} is ready for agent testing")
        print("📋 Task: 'Create an art of an enlightened male being'")
        print("🖼️ Proof: Enlightened being artwork submitted")
        print("⏳ Status: InProgress with proof - ready for agent verification")
        print("\n🤖 Now run your TaskFi agent to see it process this task!")
        print("   python taskfi_agent_fixed.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())