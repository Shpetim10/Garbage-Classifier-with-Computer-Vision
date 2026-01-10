"""
app.py
Main Streamlit application for Garbage Classification

Features:
- Upload multiple images or capture from camera
- Real-time classification with confidence scores
- Knowledge base for recyclability information
- User feedback collection
- Batch processing and ZIP download
- Loading animations
"""

import streamlit as st
import pandas as pd
from PIL import Image
import io
import time
from datetime import datetime
from pathlib import Path

# Import custom modules
from model_utils import GarbageClassifier, PredictionLogger
from knowledge_base import RecyclabilityKnowledgeBase
from image_processing import ImageProcessor, BatchProcessor

# Page configuration
st.set_page_config(
    page_title="Smart Garbage Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #ddd;
    }
    .confident {
        background-color: #E8F5E9;
        border-color: #4CAF50;
    }
    .uncertain {
        background-color: #FFF9C4;
        border-color: #FFC107;
    }
    .out-of-scope {
        background-color: #FFEBEE;
        border-color: #F44336;
    }
    .stButton>button {
        width: 100%;
        background-color: #2E7D32;
        color: white;
        padding: 0.75rem;
        font-size: 1rem;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1B5E20;
    }
    .fact-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []


@st.cache_resource
def load_classifier():
    """Load model (cached)"""
    return GarbageClassifier(
        model_path="models/garbage_classifier_transfer_learning_model_b.keras",
        confidence_threshold=0.70,
        out_of_scope_threshold=0.50
    )


@st.cache_resource
def load_knowledge_base():
    """Load knowledge base (cached)"""
    return RecyclabilityKnowledgeBase()


def display_prediction(image, prediction, filename, kb, idx):
    """
    Display prediction results for a single image
    
    Args:
        image: PIL Image
        prediction: Prediction dictionary
        filename: Image filename
        kb: Knowledge base instance
        idx: Image index
    """
    # Determine CSS class based on status
    status_class = prediction['status']
    
    # Create columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown(f"### 📁 {filename}")
        
        # Status badge
        if status_class == 'confident':
            st.success(f"✅ Confident Prediction")
        elif status_class == 'uncertain':
            st.warning(f"⚠️ Uncertain Prediction")
        else:
            st.error(f"❌ Out of Scope")
        
        # Main prediction
        if status_class == 'out_of_scope':
            st.markdown(f"""
            <div class="prediction-box out-of-scope">
                <h3>🚫 Not in Model Capabilities</h3>
                <p>This image doesn't appear to be garbage that I can classify.</p>
                <p>I work best with: batteries, organic waste, glass, cardboard, clothes, 
                metal, paper, plastic, shoes, and general trash.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("**Top 3 possibilities (for reference):**")
            for item in prediction['top_3']:
                st.write(f"{item['rank']}. **{item['class']}**: {item['confidence']:.1%}")
        
        elif status_class == 'uncertain':
            st.markdown(f"""
            <div class="prediction-box uncertain">
                <h3>⚠️ Uncertain Classification</h3>
                <p><strong>Most likely:</strong> {prediction['predicted_class']}</p>
                <p><strong>Confidence:</strong> {prediction['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("**Top 3 possibilities:**")
            for item in prediction['top_3']:
                st.write(f"{item['rank']}. **{item['class']}**: {item['confidence']:.1%}")
        
        else:  # confident
            st.markdown(f"""
            <div class="prediction-box confident">
                <h3>✅ {prediction['predicted_class'].upper()}</h3>
                <p><strong>Confidence:</strong> {prediction['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recyclability information
        if status_class != 'out_of_scope':
            with st.expander("♻️ Recyclability Information"):
                info = kb.get_info(prediction['predicted_class'])
                
                # Recyclability status
                st.markdown(f"**Status:** {kb.get_recyclability_status(prediction['predicted_class'])}")
                st.markdown(f"**Category:** {info['category']}")
                
                # Instructions
                st.markdown("**How to Recycle:**")
                for instruction in info['recycling_instructions']:
                    st.write(f"• {instruction}")
                
                # Environmental impact
                st.markdown(f"**Environmental Impact:** {info['environmental_impact']}")
                
                # Special notes
                if info['special_notes']:
                    st.info(f"💡 **Note:** {info['special_notes']}")
        
        # Feedback section
        with st.expander("📝 Provide Feedback"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                is_correct = st.radio(
                    "Was this prediction correct?",
                    ["Yes", "No", "Not sure"],
                    key=f"correct_{idx}"
                )
            
            with col_b:
                if is_correct == "No":
                    actual_class = st.selectbox(
                        "What should it be?",
                        ["Select..."] + sorted(kb.knowledge.keys()),
                        key=f"actual_{idx}"
                    )
            
            feedback_text = st.text_area(
                "Additional comments (optional):",
                key=f"feedback_{idx}",
                height=80
            )
            
            if st.button("Submit Feedback", key=f"submit_{idx}"):
                # Store feedback
                feedback_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'filename': filename,
                    'predicted_class': prediction['predicted_class'],
                    'confidence': prediction['confidence'],
                    'is_correct': is_correct,
                    'actual_class': actual_class if is_correct == "No" else None,
                    'feedback_text': feedback_text
                }
                st.session_state.feedback_data.append(feedback_entry)
                
                # Log to file
                logger = PredictionLogger()
                logger.log_prediction(
                    predicted_class=prediction['predicted_class'],
                    confidence=prediction['confidence'],
                    status=prediction['status'],
                    correct=(is_correct == "Yes"),
                    user_feedback=feedback_text
                )
                
                st.success("✅ Thank you for your feedback!")
    
    st.markdown("---")


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">♻️ Smart Garbage Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Classify waste items and learn how to recycle them properly</p>', unsafe_allow_html=True)
    
    # Load resources
    with st.spinner("🔄 Loading AI model..."):
        classifier = load_classifier()
        kb = load_knowledge_base()
        processor = ImageProcessor()
        batch_processor = BatchProcessor()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        This app uses AI to classify garbage and provide recycling information.
        
        **Supported categories:**
        - 🔋 Batteries
        - 🍂 Biological waste
        - 🥛 Glass (clear, brown, green)
        - 📦 Cardboard
        - 👕 Clothes
        - 🥫 Metal
        - 📄 Paper
        - ♻️ Plastic
        - 👟 Shoes
        - 🗑️ General trash
        """)
        
        st.header("⚙️ Settings")
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.70,
            step=0.05,
            help="Minimum confidence for single prediction"
        )
        classifier.confidence_threshold = confidence_threshold
        
        # Use TTA
        use_tta = st.checkbox(
            "Use Test-Time Augmentation",
            value=False,
            help="Slower but more accurate"
        )
        
        # Random fact
        st.header("💡 Did You Know?")
        facts = kb.get_quick_facts()
        import random
        st.markdown(f'<div class="fact-box">{random.choice(facts)}</div>', unsafe_allow_html=True)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📸 Classify Images", "📊 View Results", "🔍 Knowledge Base"])
    
    with tab1:
        st.header("Upload or Capture Images")
        
        # Upload method selection
        upload_method = st.radio(
            "Choose input method:",
            ["Upload Files", "Use Camera"],
            horizontal=True
        )
        
        uploaded_images = []
        
        if upload_method == "Upload Files":
            uploaded_files = st.file_uploader(
                "Choose images...",
                type=['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp'],
                accept_multiple_files=True,
                help="You can upload multiple images at once"
            )
            
            if uploaded_files:
                uploaded_images = uploaded_files
        
        else:  # Camera
            camera_photo = st.camera_input("Take a photo")
            if camera_photo:
                uploaded_images = [camera_photo]
        
        # Process button
        if uploaded_images:
            st.info(f"📎 {len(uploaded_images)} image(s) ready to process")
            
            if st.button("🚀 Classify Images", type="primary"):
                # Clear previous results
                st.session_state.processed_images = []
                st.session_state.predictions = []
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_images):
                    # Update progress
                    progress = (idx + 1) / len(uploaded_images)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing image {idx + 1}/{len(uploaded_images)}... 🤔")
                    
                    # Validate image
                    is_valid, error_msg = processor.validate_image(uploaded_file)
                    if not is_valid:
                        st.error(f"❌ {uploaded_file.name}: {error_msg}")
                        continue
                    
                    # Load image
                    uploaded_file.seek(0)
                    image = Image.open(uploaded_file)
                    
                    # Resize if needed
                    image = processor.resize_image(image)
                    
                    # Make prediction
                    if use_tta:
                        prediction = classifier.predict_with_tta(image, num_augmentations=5)
                    else:
                        prediction = classifier.predict(image)
                    
                    # Store results
                    st.session_state.processed_images.append((image, uploaded_file.name))
                    st.session_state.predictions.append(prediction)
                    
                    # Small delay for UX
                    time.sleep(0.1)
                
                progress_bar.empty()
                status_text.empty()
                st.success(f"✅ Processed {len(st.session_state.processed_images)} images!")
                st.balloons()
        
        # Display results
        if st.session_state.processed_images:
            st.markdown("---")
            st.header("📋 Classification Results")
            
            for idx, ((image, filename), prediction) in enumerate(zip(
                st.session_state.processed_images, 
                st.session_state.predictions
            )):
                display_prediction(image, prediction, filename, kb, idx)
    
    with tab2:
        st.header("📊 Batch Processing Results")
        
        if not st.session_state.processed_images:
            st.info("👆 Process some images first to see results here!")
        else:
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            total = len(st.session_state.predictions)
            confident = sum(1 for p in st.session_state.predictions if p['status'] == 'confident')
            uncertain = sum(1 for p in st.session_state.predictions if p['status'] == 'uncertain')
            out_of_scope = sum(1 for p in st.session_state.predictions if p['status'] == 'out_of_scope')
            
            col1.metric("Total Images", total)
            col2.metric("Confident", confident, f"{confident/total*100:.0f}%")
            col3.metric("Uncertain", uncertain, f"{uncertain/total*100:.0f}%")
            col4.metric("Out of Scope", out_of_scope, f"{out_of_scope/total*100:.0f}%")
            
            # Distribution chart
            st.subheader("Distribution by Class")
            
            class_counts = {}
            for pred in st.session_state.predictions:
                if pred['status'] != 'out_of_scope':
                    class_name = pred['predicted_class']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            if class_counts:
                df = pd.DataFrame({
                    'Class': list(class_counts.keys()),
                    'Count': list(class_counts.values())
                }).sort_values('Count', ascending=False)
                
                st.bar_chart(df.set_index('Class'))
            
            # Download section
            st.markdown("---")
            st.subheader("📦 Export Results")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                # Download ZIP
                if st.button("📥 Download ZIP (Organized by Class)", use_container_width=True):
                    with st.spinner("Creating ZIP file... 🔄"):
                        images = [img for img, _ in st.session_state.processed_images]
                        filenames = [fn for _, fn in st.session_state.processed_images]
                        
                        zip_bytes = batch_processor.create_zip_with_summary(
                            images,
                            st.session_state.predictions,
                            filenames
                        )
                        
                        st.download_button(
                            label="💾 Download classified_garbage.zip",
                            data=zip_bytes,
                            file_name="classified_garbage.zip",
                            mime="application/zip"
                        )
            
            with col_b:
                # Download CSV
                if st.button("📊 Download CSV Report", use_container_width=True):
                    # Create DataFrame
                    records = []
                    for (_, filename), pred in zip(
                        st.session_state.processed_images,
                        st.session_state.predictions
                    ):
                        records.append({
                            'Filename': filename,
                            'Predicted Class': pred['predicted_class'],
                            'Confidence': f"{pred['confidence']:.2%}",
                            'Status': pred['status']
                        })
                    
                    df = pd.DataFrame(records)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="💾 Download report.csv",
                        data=csv,
                        file_name="classification_report.csv",
                        mime="text/csv"
                    )
    
    with tab3:
        st.header("🔍 Recyclability Knowledge Base")
        
        # Search functionality
        search_query = st.text_input("🔎 Search knowledge base:", placeholder="e.g., 'plastic recycling'")
        
        if search_query:
            results = kb.search_knowledge(search_query)
            if results:
                st.success(f"Found {len(results)} result(s)")
                for result in results[:5]:
                    with st.expander(f"{result['info']['icon']} {result['item'].title()}"):
                        info = result['info']
                        st.markdown(f"**Status:** {kb.get_recyclability_status(result['item'])}")
                        st.markdown(f"**Category:** {info['category']}")
                        st.markdown("**Instructions:**")
                        for instruction in info['recycling_instructions']:
                            st.write(f"• {instruction}")
            else:
                st.warning("No results found")
        
        # Browse by category
        st.subheader("Browse by Category")
        categories = kb.get_all_categories()
        
        selected_category = st.selectbox("Select a category:", categories)
        
        if selected_category:
            items = kb.get_items_by_category(selected_category)
            st.write(f"**Items in {selected_category}:**")
            
            for item in items:
                with st.expander(f"{kb.get_info(item)['icon']} {item.title()}"):
                    info = kb.get_info(item)
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"**{kb.get_recyclability_status(item)}**")
                        env_score = kb.get_environmental_score(item)
                        st.progress(env_score / 100)
                        st.caption(f"Environmental Impact: {info['environmental_impact']}")
                    
                    with col2:
                        st.markdown("**Recycling Instructions:**")
                        for instruction in info['recycling_instructions']:
                            st.write(f"• {instruction}")
                        
                        if info['special_notes']:
                            st.info(f"💡 {info['special_notes']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🌍 Help protect our environment by recycling properly</p>
        <p style='font-size: 0.9rem;'>Developed by Artjol Zaimi, Eglis Braho and Shpëtim Shabanaj</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
