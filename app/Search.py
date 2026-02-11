import streamlit as st
from services.searching import HybridSearchService
from haystack import Document
from typing import List
from pathlib import Path

if "search_terms" not in st.session_state:
    st.session_state["search_terms"] = []

def highlight_text(text: str, query: str) -> str:
    """Highlight query terms in the text."""
    if not query:
        return text
    
    # Simple word-based highlighting
    words = query.lower().split()
    highlighted = text
    for word in words:
        if len(word) > 2:  # Only highlight words longer than 2 chars
            # Case-insensitive replacement with highlight
            import re
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted = pattern.sub(
                lambda m: f":orange-background[**{m.group()}**]", 
                highlighted
            )
    return highlighted

def render_search_results(documents: List[Document], query: str = ""):
    """
    Render Haystack documents in a nice Streamlit layout.
    
    Args:
        documents: List of Haystack Document objects
        query: Original search query for highlighting
    """
    if not documents:
        st.info("No results found.")
        return
    
    st.subheader(f"Found {len(documents)} results")
    
    for idx, doc in enumerate(documents, 1):
        # Create an expander for each result
        score = doc.score if hasattr(doc, 'score') and doc.score else 0.0
        
        # Extract metadata
        meta = doc.meta or {}
        try:
            source_file = meta['dl_meta']['meta']['origin']['filename']
        except KeyError:
            source_file = "Unknown"
        
        if isinstance(source_file, (str, Path)):
            source_name = Path(source_file).name
        else:
            source_name = str(source_file)
        
        page_num = meta.get('page_number', meta.get('page', None))
        
        # Create header with score badge
        col1, col2 = st.columns([4, 1])
        with col1:
            header = f"Result {idx}: {source_name}"
            if page_num:
                header += f" (Page {page_num})"
        with col2:
            if score > 0:
                st.metric("Score", f"{score:.3f}")
        
        with st.expander(header, expanded=(idx == 1)):
            # Metadata section
            with st.container():
                st.markdown("**📄 Document Information**")
                
                info_cols = st.columns(3)
                with info_cols[0]:
                    st.caption("Source")
                    st.text(source_name)
                
                with info_cols[1]:
                    if page_num:
                        st.caption("Page")
                        st.text(str(page_num))
                    else:
                        st.caption("Page")
                        st.text("N/A")
                
                with info_cols[2]:
                    chunk_id = meta.get('chunk_id', meta.get('id', None))
                    if chunk_id:
                        st.caption("Chunk ID")
                        st.text(str(chunk_id)[:20])
                
                # Additional metadata
                other_meta = {k: v for k, v in meta.items()}
                #             if k not in ['file_path', 'source', 'page_number', 
                #                         'page', 'chunk_id', 'id', '_split_id']}
                if other_meta:
                    with st.expander("Additional metadata"):
                        st.json(other_meta)
            
            st.divider()
            
            # Content section
            st.markdown("**📝 Content**")
            
            # Highlight matching text
            content = doc.content or ""
            if query:
                highlighted_content = highlight_text(content, query)
                st.markdown(highlighted_content)
            else:
                st.markdown(content)
            
            # Show content length
            st.caption(f"Content length: {len(content)} characters")
            
            # Action buttons
            st.markdown("")
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])
            
            with btn_col1:
                if st.button("📋 Copy", key=f"copy_{idx}"):
                    st.code(content, language=None)
            
            with btn_col2:
                if st.button("🔗 ID", key=f"id_{idx}"):
                    st.code(doc.id, language=None)

def add_search_term(term, polarity, value):
    if "search_terms" not in st.session_state:
        st.session_state["search_terms"] = []
    st.session_state["search_terms"].append(
        {
            "term": term,
            "polarity": polarity,
            "value": value,
        }
    )
    return
    


st.title("Hybrid Search")

st.write("Search among the indexed documents.")

keyword_weight = st.slider("Keyword Weight", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
semantic_weight = 1.0 - keyword_weight

#query = st.text_input(placeholder="Enter your search query here...", label="Search Query")



term = st.text_input(placeholder="Diagnose", label="Search term")

polarity = st.selectbox(options=["=", "!="], label="Polarity")

value = st.text_input(placeholder="Diabetes mellitus Typ 2", label="Value term")

add_search_term_button = st.button(label="Add search term")

if add_search_term_button:
    if not (term and value):
        st.error("Enter search term and value!")
    add_search_term(term, polarity, value)

#TODO: prettify
st.write(f"### Current Search Terms: {st.session_state['search_terms']}")

search_button = st.button(label="Search")

if search_button:
    st.write(f"You searched for: {st.session_state['search_terms']}")
    search_service = HybridSearchService(keyword_weight=keyword_weight, semantic_weight=semantic_weight)
    with st.spinner("Searching..."):
        results = search_service.search(search_terms=st.session_state["search_terms"])
    st.write("Search Results:")
    #st.write(results)
    
    render_search_results(results, None)