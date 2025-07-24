import streamlit as st
import asyncio
import logging
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List

from research_agent import ResearchAgent
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS that adapts to light/dark mode
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        color: var(--primary-color, #1f77b4);
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Query box - adapts to theme */
    .query-box {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid rgba(49, 51, 63, 0.2);
    }
    
    /* Answer box - adapts to theme */
    .answer-box {
        background-color: var(--background-color);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid var(--primary-color, #1f77b4);
        margin-bottom: 1rem;
        border: 1px solid rgba(49, 51, 63, 0.2);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Source box - adapts to theme */
    .source-box {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #28a745;
        border: 1px solid rgba(49, 51, 63, 0.2);
    }
    
    /* Statistics box - adapts to theme */
    .stat-box {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid rgba(49, 51, 63, 0.2);
    }
    
    /* Error box - works in both themes */
    .error-box {
        background-color: rgba(248, 215, 218, 0.3);
        color: #d32f2f;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(245, 198, 203, 0.5);
    }
    
    /* Dark mode specific overrides */
    [data-theme="dark"] .query-box,
    [data-theme="dark"] .answer-box,
    [data-theme="dark"] .source-box,
    [data-theme="dark"] .stat-box {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
    }
    
    /* Ensure text is visible in both modes */
    .answer-box, .source-box, .stat-box, .query-box {
        color: var(--text-color);
    }
    
    /* Link styling for both themes */
    .source-box a {
        color: var(--primary-color, #1f77b4);
        text-decoration: none;
    }
    
    .source-box a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitApp:
    def __init__(self):
        self.agent = None
        self.initialize_agent()
    
    def initialize_agent(self):
        """Initialize the research agent"""
        try:
            if 'agent' not in st.session_state:
                with st.spinner("Initializing research agent..."):
                    st.session_state.agent = ResearchAgent()
            self.agent = st.session_state.agent
        except Exception as e:
            st.error(f"Failed to initialize research agent: {e}")
            st.stop()
    
    def run(self):
        """Main application function"""
        # Header
        st.markdown(f'<h1 class="main-header">{config.APP_TITLE}</h1>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align: center; opacity: 0.7;">{config.APP_DESCRIPTION}</p>', unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        self.render_main_content()
    
    def render_sidebar(self):
        """Render the sidebar with stats and controls"""
        st.sidebar.header("📊 System Status")
        
        # Vector store stats
        try:
            stats = self.agent.get_vector_store_stats()
            
            st.sidebar.metric("Total Chucks in vDB", stats['total_chunks'])
            st.sidebar.metric("Unique Papers", stats['unique_papers'])
            
            # Show papers in vector store
            if stats['papers']:
                st.sidebar.subheader("📑 Papers in Database")
                for paper in stats['papers'][:5]:  # Show first 5
                    st.sidebar.text(f"• {paper[:50]}...")
                
                if len(stats['papers']) > 5:
                    st.sidebar.text(f"... and {len(stats['papers']) - 5} more")
            
        except Exception as e:
            st.sidebar.error(f"Error loading stats: {e}")
        
        # Controls
        st.sidebar.header("🔧 Controls")
        
        # Clear vector store with confirmation
        if 'confirm_clear' not in st.session_state:
            st.session_state.confirm_clear = False
        
        if not st.session_state.confirm_clear:
            if st.sidebar.button("Clear Vector Store", type="secondary"):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.sidebar.warning("⚠️ Are you sure you want to clear all stored documents?")
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("✅ Yes", key="confirm_yes"):
                    try:
                        success = self.agent.clear_vector_store()
                        if success:
                            st.sidebar.success("Vector store cleared!")
                            st.session_state.confirm_clear = False
                            st.rerun()
                        else:
                            st.sidebar.error("Failed to clear vector store")
                            st.session_state.confirm_clear = False
                    except Exception as e:
                        st.sidebar.error(f"Error: {e}")
                        st.session_state.confirm_clear = False
            with col2:
                if st.button("❌ No", key="confirm_no"):
                    st.session_state.confirm_clear = False
                    st.rerun()
        
        # Configuration
        st.sidebar.header("⚙️ Configuration")
        st.sidebar.text(f"Model: {config.GEMINI_MODEL}")
        st.sidebar.text(f"Max Results: {config.MAX_RESULTS}")
        st.sidebar.text(f"Temperature: {config.TEMPERATURE}")
    
    def render_main_content(self):
        """Render the main content area"""
        # Query input
        st.subheader("🔍 Ask a Research Question")
        
        # Example queries
        with st.expander("💡 Example Queries"):
            st.write("""
            - "What are the latest developments in transformer architectures?"
            - "How does climate change affect biodiversity in marine ecosystems?"
            - "What are the most effective methods for few-shot learning?"
            - "Recent advances in quantum computing error correction"
            - "Machine learning applications in drug discovery"
            """)
        
        # Query input form
        with st.form("query_form"):
            query = st.text_area(
                "Enter your research question:",
                height=100,
                placeholder="e.g., What are the latest developments in transformer architectures for natural language processing?"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                submit_button = st.form_submit_button("🔍 Search", type="primary")
            with col2:
                if st.form_submit_button("🧹 Clear History"):
                    st.session_state.clear()
                    st.rerun()
        
        # Process query
        if submit_button and query.strip():
            self.process_query(query)
        
        # Display chat history
        self.display_chat_history()
    
    def process_query(self, query: str):
        """Process a research query"""
        # Add to session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Show query
        st.session_state.chat_history.append({
            'type': 'user',
            'content': query,
            'timestamp': datetime.now()
        })
        
        # Process with agent
        with st.spinner("🔍 Searching for relevant papers..."):
            try:
                # Run async function
                result = asyncio.run(self.agent.research(query))
                
                # Add result to history
                st.session_state.chat_history.append({
                    'type': 'assistant',
                    'content': result,
                    'timestamp': datetime.now()
                })
                
                # Rerun to display results
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
                st.session_state.chat_history.append({
                    'type': 'error',
                    'content': str(e),
                    'timestamp': datetime.now()
                })
    
    def display_chat_history(self):
        """Display chat history"""
        if 'chat_history' not in st.session_state:
            return
        
        st.subheader("💬 Research Results")
        
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            if message['type'] == 'user':
                self.display_user_message(message)
            elif message['type'] == 'assistant':
                self.display_assistant_message(message)
            elif message['type'] == 'error':
                self.display_error_message(message)
            
            # Add separator
            if i < len(st.session_state.chat_history) - 1:
                st.divider()
    
    def display_user_message(self, message: Dict):
        """Display user message"""
        st.markdown(f"""
        <div class="query-box">
            <strong>🤔 Your Question:</strong><br>
            {message['content']}
            <br><small>Asked at {message['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small>
        </div>
        """, unsafe_allow_html=True)
    
    def display_assistant_message(self, message: Dict):
        """Display assistant message with research results"""
        result = message['content']
        
        # Main answer
        st.markdown(f"""
        <div class="answer-box">
            <strong>🤖 Research Assistant:</strong><br>
            {result['answer']}
        </div>
        """, unsafe_allow_html=True)
        
        # Sources and statistics
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Sources
            if result['sources']:
                st.subheader("📚 Sources")
                for source in result['sources']:
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>{source['title']}</strong><br>
                        <small>Authors: {self._format_authors(source['authors'])}</small><br>
                        <small>Year: {source['year'] or 'Unknown'} | Citations: {source['citation_count']}</small><br>
                        <a href="{source['url']}" target="_blank">View Paper</a>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            # Statistics
            st.subheader("📊 Query Statistics")
            
            st.markdown(f"""
            <div class="stat-box">
                <strong>Papers Found:</strong> {len(result['sources'])}<br>
                <strong>Processed:</strong> {len(result['processed_papers'])}<br>
                <strong>Context Chunks:</strong> {result['context_chunks_count']}<br>
                <strong>Status:</strong> {result['status']}<br>
            </div>
            """, unsafe_allow_html=True)
            
            # Processed papers
            if result['processed_papers']:
                st.subheader("✅ Processed Papers")
                for paper in result['processed_papers']:
                    st.text(f"• {paper[:40]}...")
        
        # Visualization
        if result['sources']:
            self.create_visualizations(result)
        
        # Timestamp
        st.caption(f"Response generated at {message['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    def display_error_message(self, message: Dict):
        """Display error message"""
        st.markdown(f"""
        <div class="error-box">
            <strong>❌ Error:</strong><br>
            {message['content']}
            <br><small>Occurred at {message['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small>
        </div>
        """, unsafe_allow_html=True)
    
    def create_visualizations(self, result: Dict):
        """Create visualizations for research results"""
        sources = result['sources']
        
        if not sources:
            return
        
        st.subheader("📈 Research Insights")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📅 Publication Timeline", "📊 Citation Analysis", "👥 Author Network"])
        
        with tab1:
            # Publication timeline
            years = [s['year'] for s in sources if s['year']]
            if years:
                year_counts = pd.Series(years).value_counts().sort_index()
                
                fig = px.line(
                    x=year_counts.index,
                    y=year_counts.values,
                    title="Publications by Year",
                    labels={'x': 'Year', 'y': 'Number of Papers'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Citation analysis
            citations = [s['citation_count'] for s in sources if s['citation_count'] > 0]
            if citations:
                fig = px.histogram(
                    x=citations,
                    title="Citation Distribution",
                    labels={'x': 'Citation Count', 'y': 'Number of Papers'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Author network (simplified)
            all_authors = []
            for source in sources:
                if source['authors']:
                    formatted_authors = self._format_authors_list(source['authors'])
                    all_authors.extend(formatted_authors[:3])  # First 3 authors
            
            if all_authors:
                author_counts = pd.Series(all_authors).value_counts().head(10)
                
                fig = px.bar(
                    x=author_counts.values,
                    y=author_counts.index,
                    orientation='h',
                    title="Top Authors",
                    labels={'x': 'Number of Papers', 'y': 'Author'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _format_authors(self, authors):
        """Format authors field to handle both string and dict formats"""
        if not authors:
            return 'Unknown'
        
        if isinstance(authors, list):
            # Handle list of authors
            formatted_authors = []
            for author in authors[:3]:  # Show only first 3 authors
                if isinstance(author, dict):
                    # Extract name from dict (common in scholarly library)
                    name = author.get('name', author.get('title', str(author)))
                    formatted_authors.append(name)
                elif isinstance(author, str):
                    formatted_authors.append(author)
                else:
                    formatted_authors.append(str(author))
            
            result = ', '.join(formatted_authors)
            if len(authors) > 3:
                result += f' et al. ({len(authors)} total)'
            return result
        
        elif isinstance(authors, str):
            return authors
        
        else:
            return str(authors)
    
    def _format_authors_list(self, authors):
        """Format authors for use in lists (returns list of strings)"""
        if not authors:
            return []
        
        if isinstance(authors, list):
            formatted_authors = []
            for author in authors:
                if isinstance(author, dict):
                    name = author.get('name', author.get('title', str(author)))
                    formatted_authors.append(name)
                elif isinstance(author, str):
                    formatted_authors.append(author)
                else:
                    formatted_authors.append(str(author))
            return formatted_authors
        
        elif isinstance(authors, str):
            return [authors]
        
        else:
            return [str(authors)]

def main():
    """Main function to run the Streamlit app"""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()