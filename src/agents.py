"""Multi-agent refund processing system using LangGraph."""

from typing import Dict, List
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages import convert_to_messages
from .model import get_openai_model

from .db_tools import (
    get_customer_info, get_order_info, get_product_info, get_customer_orders,
    save_processed_request, search_orders_by_customer_email,
    check_previous_refund_requests
)
from .vector_db_tools import (
    search_return_eligibility_policy, search_refund_calculation_policy,
    search_customer_tier_benefits, search_exception_handling_policy,
    search_general_policy
)
from .email_tools import send_email_notification, log_communication, generate_request_id


def pretty_print_message(message, indent=False):
    """Pretty print a single message with optional indentation."""
    pretty_message = message.pretty_repr(html=False)  # Use text format for terminal
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    """Pretty print messages from LangGraph updates."""
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print()
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print()

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print()


class RefundProcessingSystem:
    """Multi-agent refund processing system with LangGraph supervisor."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the multi-agent system."""
        self.llm = get_openai_model(config_path)
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(return_messages=True)
        self.conversation_history = []
        
        # Create agent tools
        self.validation_tools = [
            get_customer_info, get_order_info, get_product_info, get_customer_orders,
            search_orders_by_customer_email, check_previous_refund_requests
        ]
        self.policy_tools = [
            search_return_eligibility_policy, search_refund_calculation_policy,
            search_customer_tier_benefits, search_exception_handling_policy,
            search_general_policy
        ]
        self.communication_tools = [
            send_email_notification, log_communication, save_processed_request,
          generate_request_id
        ]
        
        # Create specialized agents
        self.validation_agent = self._create_validation_agent()
        self.policy_agent = self._create_policy_agent()
        self.communication_agent = self._create_communication_agent()
        
        # Create supervisor
        self.supervisor = self._create_supervisor()
        
    def _create_validation_agent(self):
        """Create validation agent for customer/order verification."""
        return create_react_agent(
            model=self.llm,
            tools=self.validation_tools,
            prompt=(
                "You are a Validation Agent for a refund/return processing system.\n\n"
                "CRITICAL: You MUST use the available tools to get real data. DO NOT make up or guess any information.\n\n"
                "AVAILABLE TOOLS:\n"
                "- get_customer_info: Look up customer by email address\n"
                "- search_orders_by_customer_email: Find orders for a customer\n"
                "- get_customer_info: Get customer details by customer ID\n"
                "- get_order_info: Get order details by order ID\n"
                "- get_product_info: Get product details by product ID\n"
                "- get_customer_orders: Get customer's order history\n"
                "- check_previous_refund_requests: Check if customer already requested refund for this order\n\n"
                "WORKFLOW:\n"
                "1. If you have an email but no customer info: Use get_customer_info\n"
                "2. If customer wants to return, you need to show orders: Use search_orders_by_customer_email\n"
                "3. If you have order ID but need details: Use get_order_info\n"
                "4. If you have product ID but need details: Use get_product_info\n"
                "5. For refund requests: Use check_previous_refund_requests for duplicate checking\n\n"
                "STRICT RULES:\n"
                "- NEVER make up customer data, order IDs, or product information\n"
                "- ALWAYS call the appropriate tool to get real data\n"
                "- If a tool returns 'found: false', report that the item was not found\n"
                "- ALWAYS check for previous refund requests to prevent duplicates\n"
                "- If check_previous_refund_requests returns 'blocking: true', inform supervisor that customer already has approved refunds\n"
                "- Only provide information that comes directly from tool results\n"
                "- After using tools, provide clear validation results based on the real data\n"
                "- Always provide all the details to supervisor about orders - order id , product id category product name etc. Use appropriate tools to find all info and compile and then provide to supervisor."
            ),
            name="validation_agent"
        )
    
    def _create_policy_agent(self):
        """Create policy agent for decision making."""
        return create_react_agent(
            model=self.llm,
            tools=self.policy_tools,
            prompt=(
                "You are a Policy Agent for refund/return processing.\n\n"
                "CRITICAL: You MUST use the available tools to search for actual policies. DO NOT make assumptions.\n\n"
                "AVAILABLE TOOLS:\n"
                "- search_return_eligibility_policy: Search return window policies by product category\n"
                "- search_refund_calculation_policy: Search refund calculation rules\n"
                "- search_customer_tier_benefits: Search tier-specific customer benefits\n"
                "- search_exception_handling_policy: Search exception handling procedures\n"
                "- search_general_policy: Search general policies by query\n\n"
                "CRITICAL: When receiving delegation from supervisor, extract actual values from conversation:\n"
                "- Look for REAL product category (e.g., 'Electronics', 'Clothing', NOT 'general')\n"
                "- Look for REAL customer tier (e.g., 'Premium', 'Gold', 'Bronze', NOT 'standard')\n"
                "- If you don't see these values, ask for them explicitly\n\n"
                "WORKFLOW:\n"
                "1. Use search_return_eligibility_policy with ACTUAL product category\n"
                "2. Use search_customer_tier_benefits with ACTUAL customer tier\n"
                "3. Use search_refund_calculation_policy with ACTUAL product category\n"
                "4. Use search_exception_handling_policy if special circumstances apply\n\n"
                "DECISION MAKING:\n"
                "- Approved: Customer eligible, within policy guidelines\n"
                "- Rejected: Outside return window or violates policies\n"
                "- Needs Review: Complex case requiring human intervention\n\n"
                "STRICT RULES:\n"
                "- ALWAYS search for relevant policies using tools before making decisions\n"
                "- Base ALL decisions on actual policy search results, not assumptions\n"
                "- Provide specific policy citations in your reasoning\n"
                "- Calculate exact refund amounts based on found policies\n"
            ),
            name="policy_agent"
        )
    
    def _create_communication_agent(self):
        """Create communication agent for customer notifications."""
        return create_react_agent(
            model=self.llm,
            tools=self.communication_tools,
            prompt=(
                "You are a Communication Agent for customer notifications and record keeping.\n\n"
                "CRITICAL: You MUST use the available tools to actually send emails and save data.\n\n"
                "AVAILABLE TOOLS:\n"
                "- send_email_notification: Send email to customer with subject and message\n"
                "- log_communication: Log communication activities for audit trail\n"
                "- save_processed_request: Save complete request processing results to database\n\n"
                "WORKFLOW:\n"
                "1. Use send_email_notification to notify customer of decision\n"
                "2. Use log_communication to record the communication\n"
                "3. Use generate_request_id tool to first generate a request id and then use that request id to pass in next tool, save_processed_request."
                "3. Use save_processed_request to save all processing details to database\n\n"
                "EMAIL GUIDELINES:\n"
                "- Use professional, empathetic tone\n"
                "- Include request ID and relevant details\n"
                "- Provide clear next steps for customer\n"
                "- Include contact information for follow-up\n\n"
                "STRICT RULES:\n"
                "- ALWAYS use tools to actually send emails and save data\n"
                "- DO NOT just say you will send an email - actually use the tools\n"
                "- Provide confirmation of actions taken using tool results\n"
            ),
            name="communication_agent"
        )
    
    def _create_supervisor(self):
        """Create supervisor to orchestrate the agents."""
        return create_supervisor(
            model=self.llm,
            agents=[self.validation_agent, self.policy_agent, self.communication_agent],
            prompt=(
                "You are a friendly customer service representative helping customers with refunds, returns, and exchanges.\n\n"
                "CRITICAL: Your agents MUST use their tools to get real data. Do NOT accept hallucinated information.\n\n"
                "CONVERSATION MEMORY:\n"
                "- You have access to the full conversation history\n"
                "- Remember what you've already learned: customer email, orders, issues discussed\n"
                "- DON'T ask for information you already have\n"
                "- Reference previous parts of the conversation naturally\n\n"
                "AGENT DELEGATION (only do ONE step per response):\n"
                "1. If no email provided yet: Ask for email address\n"
                "2. If email provided but customer not looked up: Delegate to VALIDATION AGENT with explicit instruction: 'Use get_customer_info tool to look up customer with email [email]'\n"
                "3. If customer found but orders not shown: Delegate to VALIDATION AGENT with explicit instruction: 'Use search_orders_by_customer_email tool to find orders for [email]'\n"
                "4. If order mentioned but issue not clear: Ask what specific problem they're having\n"
                "5. If issue described but order details needed: Delegate to VALIDATION AGENT with explicit instruction: 'Use get_order_info tool for order [order_id] and get_product_info tool for the product'\n"
                "6. If order verified but policy not checked: Delegate to POLICY AGENT with explicit instruction: 'Use policy search tools to find out relevant policies' and give all order details - order id , product category etc to policy agent\n"
                "7. If decision made but not confirmed: Tell customer the decision and ask if they want to proceed\n"
                "8. If customer confirms: Delegate to COMMUNICATION AGENT with explicit instruction: 'Use tools to send email notification and save request to database'\n\n"
                "AVAILABLE AGENTS:\n"
                "1. VALIDATION AGENT: Has tools for customer lookup, order verification, product info\n"
                "2. POLICY AGENT: Has tools for policy search, eligibility checking, refund calculations\n"
                "3. COMMUNICATION AGENT: Has tools for email sending and database storage\n\n"
                "CRITICAL RULES:\n"
                "- When delegating, give EXPLICIT tool usage instructions to agents\n"
                "- If an agent returns made-up data instead of using tools, redirect them with specific tool instructions\n"
                "- REMEMBER conversation history - don't repeat questions\n"
                "- Do only ONE step per response\n"
                "- Verify agents actually used tools by checking their responses for tool call results\n"
                "- Always trigger communication agent at the end to first email the customer of the interaction and finally update the database\n"
                "- When you find out all order details , always mention all orders and then ask which orders the customer is referring to. Do not assume which the order id."
            ),
            add_handoff_back_messages=True,
            output_mode="full_history"
        ).compile()
    
    def process_refund_request(self, request_data: Dict) -> Dict:
        """
        Process a refund/return request through the multi-agent workflow.
        
        Args:
            request_data: Dict containing request details
            
        Returns:
            Dict with processing results
        """
        request_message = f"""
        New refund request to process:
        
        Request ID: {request_data['request_id']}
        Customer ID: {request_data.get('customer_id', 'Not provided')}
        Order ID: {request_data.get('order_id', 'Not provided')}
        Request Type: {request_data['request_type']}
        Reason: {request_data['reason']}
        Description: {request_data['description']}
        
        Please process this request through the complete workflow:
        1. Start with validation to verify all details
        2. If valid, apply policies to make a decision
        3. Finally, communicate results to customer and save to database
        """
        
        try:
            # Execute the supervisor workflow
            result = self.supervisor.invoke({
                "messages": [{"role": "user", "content": request_message}]
            })
            
            # Extract the final message from supervisor
            final_messages = result.get("supervisor", {}).get("messages", [])
            if final_messages:
                final_response = final_messages[-1].content
                
                return {
                    "request_id": request_data['request_id'],
                    "status": "processed",
                    "workflow_complete": True,
                    "final_response": final_response,
                    "success": True
                }
            else:
                return {
                    "request_id": request_data['request_id'],
                    "status": "error",
                    "error": "No response from supervisor",
                    "success": False
                }
                
        except Exception as e:
            return {
                "request_id": request_data['request_id'],
                "status": "error",
                "error": str(e),
                "success": False
            }
    
    def chat_with_supervisor(self, message: str) -> str:
        """
        Chat directly with the supervisor for general inquiries.
        
        Args:
            message: User message/question
            
        Returns:
            Supervisor response
        """
        try:
            print(f"\n{'='*60}")
            print(f"USER MESSAGE: {message}")
            print(f"CONVERSATION HISTORY LENGTH: {len(self.conversation_history)}")
            print(f"{'='*60}")
            
            # Add user message to history
            self.conversation_history.append(HumanMessage(content=message))
            
            # Create messages payload with full conversation history
            messages_payload = []
            for msg in self.conversation_history:
                if isinstance(msg, HumanMessage):
                    messages_payload.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages_payload.append({"role": "assistant", "content": msg.content})
            
            print(f"SENDING {len(messages_payload)} MESSAGES TO SUPERVISOR")
            
            # Use streaming to get detailed agent execution info
            response_content = ""
            print("\n" + "="*60)
            print("AGENT EXECUTION TRACE:")
            print("="*60)
            
            for chunk in self.supervisor.stream({"messages": messages_payload}):
                pretty_print_messages(chunk)
                
                # Extract the final response from the last chunk
                if isinstance(chunk, dict):
                    for node_name, node_update in chunk.items():
                        if "messages" in node_update and node_update["messages"]:
                            last_message = node_update["messages"][-1]
                            if hasattr(last_message, 'content'):
                                response_content = last_message.content
                            elif isinstance(last_message, dict) and "content" in last_message:
                                response_content = last_message["content"]
            
            print("="*60)
            print(f"FINAL RESPONSE: {response_content}")
            print("="*60)
            
            # Add supervisor response to history
            if response_content:
                self.conversation_history.append(AIMessage(content=response_content))
            
            print(f"UPDATED CONVERSATION HISTORY LENGTH: {len(self.conversation_history)}")
            
            return response_content if response_content else "I apologize, but I couldn't process your request. Please try again."
                
        except Exception as e:
            print(f"SUPERVISOR ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"I encountered an error: {str(e)}. Please try again or contact support."
    
    def get_workflow_graph(self):
        """Get visual representation of the workflow."""
        return self.supervisor.get_graph().draw_mermaid_png()


# Global instance for easy import
refund_system = None

def get_refund_system(config_path: str = "config.yaml"):
    """Get or create global refund system instance."""
    global refund_system
    if refund_system is None:
        refund_system = RefundProcessingSystem(config_path)
    return refund_system