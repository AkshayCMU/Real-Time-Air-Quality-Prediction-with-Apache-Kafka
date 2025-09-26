"""
Kafka KRaft Setup Script
Phase 1: Streaming Infrastructure and Data Pipeline Architecture

This script sets up Apache Kafka with KRaft (Kafka Raft) mode,
eliminating the need for Zookeeper and providing a more efficient
and simpler deployment.
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KafkaKraftSetup:
    """
    Setup and configuration manager for Kafka with KRaft mode.
    """
    
    def __init__(self, kafka_home: str = None, data_dir: str = None):
        """
        Initialize Kafka KRaft setup.
        
        Args:
            kafka_home: Path to Kafka installation directory
            data_dir: Directory for Kafka data storage
        """
        self.system = platform.system().lower()
        self.kafka_home = kafka_home or self._get_default_kafka_home()
        self.data_dir = data_dir or str(Path.cwd() / "kafka_data")
        self.cluster_id = "air-quality-cluster"
        
        # Create data directory
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Kafka KRaft setup initialized")
        logger.info(f"System: {self.system}")
        logger.info(f"Kafka home: {self.kafka_home}")
        logger.info(f"Data directory: {self.data_dir}")
    
    def _get_default_kafka_home(self) -> str:
        """Get default Kafka installation path."""
        if self.system == "windows":
            return "C:\\kafka"
        else:
            return "/opt/kafka"
    
    def check_kafka_installation(self) -> bool:
        """
        Check if Kafka is properly installed.
        
        Returns:
            True if Kafka is installed, False otherwise
        """
        try:
            kafka_script = self._get_kafka_script("kafka-server-start")
            result = subprocess.run([kafka_script, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logger.info("Kafka installation found")
                logger.info(f"Version: {result.stdout.strip()}")
                return True
            else:
                logger.error("Kafka installation not found or invalid")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.error(f"Error checking Kafka installation: {e}")
            return False
    
    def _get_kafka_script(self, script_name: str) -> str:
        """Get path to Kafka script."""
        if self.system == "windows":
            return os.path.join(self.kafka_home, "bin", "windows", f"{script_name}.bat")
        else:
            return os.path.join(self.kafka_home, "bin", f"{script_name}.sh")
    
    def generate_cluster_id(self) -> str:
        """
        Generate a unique cluster ID for KRaft mode.
        
        Returns:
            Generated cluster ID
        """
        try:
            kafka_script = self._get_kafka_script("kafka-storage")
            result = subprocess.run([
                kafka_script, "random-uuid"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                cluster_id = result.stdout.strip()
                logger.info(f"Generated cluster ID: {cluster_id}")
                return cluster_id
            else:
                logger.warning("Failed to generate cluster ID, using default")
                return self.cluster_id
                
        except Exception as e:
            logger.warning(f"Error generating cluster ID: {e}, using default")
            return self.cluster_id
    
    def create_server_properties(self) -> str:
        """
        Create server.properties file for KRaft mode.
        
        Returns:
            Path to created properties file
        """
        properties_content = f"""# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# KRaft mode configuration
process.roles=broker,controller
node.id=1
controller.quorum.voters=1@localhost:9093
listeners=PLAINTEXT://localhost:9092,CONTROLLER://localhost:9093
inter.broker.listener.name=PLAINTEXT
advertised.listeners=PLAINTEXT://localhost:9092
controller.listener.names=CONTROLLER
listener.security.protocol.map=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
num.network.threads=3
num.io.threads=8
socket.send.buffer.bytes=102400
socket.receive.buffer.bytes=102400
socket.request.max.bytes=104857600
log.dirs={self.data_dir}/kafka-logs
num.partitions=3
num.recovery.threads.per.data.dir=1
offsets.topic.replication.factor=1
transaction.state.log.replication.factor=1
transaction.state.log.min.isr=1
log.retention.hours=168
log.segment.bytes=1073741824
log.retention.check.interval.ms=300000
zookeeper.connect=localhost:2181
zookeeper.connection.timeout.ms=18000
group.initial.rebalance.delay.ms=0
"""
        
        properties_file = Path(self.data_dir) / "server.properties"
        with open(properties_file, 'w') as f:
            f.write(properties_content)
        
        logger.info(f"Server properties created: {properties_file}")
        return str(properties_file)
    
    def format_storage(self) -> bool:
        """
        Format Kafka storage for KRaft mode.
        
        Returns:
            True if formatting successful, False otherwise
        """
        try:
            cluster_id = self.generate_cluster_id()
            kafka_script = self._get_kafka_script("kafka-storage")
            
            result = subprocess.run([
                kafka_script, "format",
                "-t", cluster_id,
                "-c", str(Path(self.data_dir) / "server.properties")
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("Kafka storage formatted successfully")
                return True
            else:
                logger.error(f"Failed to format storage: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error formatting storage: {e}")
            return False
    
    def start_kafka_server(self) -> subprocess.Popen:
        """
        Start Kafka server in KRaft mode.
        
        Returns:
            Process object for the running Kafka server
        """
        try:
            kafka_script = self._get_kafka_script("kafka-server-start")
            properties_file = str(Path(self.data_dir) / "server.properties")
            
            # Start Kafka server
            process = subprocess.Popen([
                kafka_script, properties_file
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Wait a moment for startup
            time.sleep(5)
            
            # Check if process is still running
            if process.poll() is None:
                logger.info("Kafka server started successfully")
                return process
            else:
                stdout, stderr = process.communicate()
                logger.error(f"Kafka server failed to start: {stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error starting Kafka server: {e}")
            return None
    
    def create_topic(self, topic_name: str = "air-quality-data", 
                    partitions: int = 3, replication_factor: int = 1) -> bool:
        """
        Create a Kafka topic.
        
        Args:
            topic_name: Name of the topic to create
            partitions: Number of partitions
            replication_factor: Replication factor
            
        Returns:
            True if topic created successfully, False otherwise
        """
        try:
            kafka_script = self._get_kafka_script("kafka-topics")
            
            result = subprocess.run([
                kafka_script, "--create",
                "--topic", topic_name,
                "--bootstrap-server", "localhost:9092",
                "--partitions", str(partitions),
                "--replication-factor", str(replication_factor)
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"Topic '{topic_name}' created successfully")
                return True
            else:
                if "already exists" in result.stderr:
                    logger.info(f"Topic '{topic_name}' already exists")
                    return True
                else:
                    logger.error(f"Failed to create topic: {result.stderr}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error creating topic: {e}")
            return False
    
    def list_topics(self) -> list:
        """
        List all Kafka topics.
        
        Returns:
            List of topic names
        """
        try:
            kafka_script = self._get_kafka_script("kafka-topics")
            
            result = subprocess.run([
                kafka_script, "--list",
                "--bootstrap-server", "localhost:9092"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                topics = [topic.strip() for topic in result.stdout.strip().split('\n') if topic.strip()]
                logger.info(f"Found topics: {topics}")
                return topics
            else:
                logger.error(f"Failed to list topics: {result.stderr}")
                return []
                
        except Exception as e:
            logger.error(f"Error listing topics: {e}")
            return []
    
    def setup_complete_environment(self) -> bool:
        """
        Set up complete Kafka KRaft environment.
        
        Returns:
            True if setup successful, False otherwise
        """
        logger.info("Starting complete Kafka KRaft setup...")
        
        # Check Kafka installation
        if not self.check_kafka_installation():
            logger.error("Kafka installation not found. Please install Kafka first.")
            return False
        
        # Create server properties
        self.create_server_properties()
        
        # Format storage
        if not self.format_storage():
            logger.error("Failed to format storage")
            return False
        
        # Start Kafka server
        kafka_process = self.start_kafka_server()
        if not kafka_process:
            logger.error("Failed to start Kafka server")
            return False
        
        # Wait for server to be ready
        logger.info("Waiting for Kafka server to be ready...")
        time.sleep(10)
        
        # Create topic
        if not self.create_topic():
            logger.error("Failed to create topic")
            return False
        
        # List topics to verify
        topics = self.list_topics()
        if "air-quality-data" not in topics:
            logger.error("Topic creation verification failed")
            return False
        
        logger.info("Kafka KRaft setup completed successfully!")
        logger.info("Kafka server is running in the background")
        logger.info("You can now run the producer and consumer applications")
        
        return True


def main():
    """Main function to run Kafka KRaft setup."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka KRaft Setup')
    parser.add_argument('--kafka-home', default=None,
                       help='Path to Kafka installation directory')
    parser.add_argument('--data-dir', default=None,
                       help='Directory for Kafka data storage')
    parser.add_argument('--topic', default='air-quality-data',
                       help='Topic name to create')
    parser.add_argument('--partitions', type=int, default=3,
                       help='Number of partitions for the topic')
    parser.add_argument('--replication-factor', type=int, default=1,
                       help='Replication factor for the topic')
    
    args = parser.parse_args()
    
    try:
        # Create setup instance
        setup = KafkaKraftSetup(
            kafka_home=args.kafka_home,
            data_dir=args.data_dir
        )
        
        # Run complete setup
        success = setup.setup_complete_environment()
        
        if success:
            logger.info("Setup completed successfully!")
            sys.exit(0)
        else:
            logger.error("Setup failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Setup error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
