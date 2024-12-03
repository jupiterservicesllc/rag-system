from fpdf import FPDF

def create_test_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add some test content
    pdf.cell(200, 10, txt="AI and Machine Learning Overview", ln=1, align='C')
    pdf.ln(10)
    
    content = """
    Artificial Intelligence (AI) and Machine Learning (ML) are transforming industries across the globe. Here are some key concepts:

    1. Machine Learning:
    - Supervised Learning: Models learn from labeled data
    - Unsupervised Learning: Models find patterns in unlabeled data
    - Reinforcement Learning: Models learn through trial and error

    2. Deep Learning:
    - Neural Networks: Inspired by human brain structure
    - Convolutional Neural Networks (CNN): Excellent for image processing
    - Recurrent Neural Networks (RNN): Great for sequential data

    3. Natural Language Processing:
    - Text Classification
    - Named Entity Recognition
    - Machine Translation
    - Question Answering Systems

    4. Applications:
    - Healthcare: Disease diagnosis and treatment planning
    - Finance: Fraud detection and risk assessment
    - Transportation: Autonomous vehicles
    - Education: Personalized learning systems
    """
    
    pdf.multi_cell(0, 10, txt=content)
    pdf.output("test_docs/ai_overview.pdf")

if __name__ == "__main__":
    import os
    os.makedirs("test_docs", exist_ok=True)
    create_test_pdf()
