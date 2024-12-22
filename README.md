# AES_using_BERT

This project implements an **Automated Essay Grading System** that uses machine learning models to evaluate essays. The system provides predictions on various aspects of the essay, including grammar, lexical richness, organization, and the quality of supporting ideas. The project utilizes a combination of models, including BERT, self-made functions, and datasets like CELA to generate comprehensive feedback for submitted essays.

## Features
- **Essay Submission**: Users can submit essays through a form interface.
- **Model Predictions**: Two models analyze the essay, providing scores on:
  - Grammar
  - Lexical richness
  - Global and local organization
  - Supporting ideas
  - Holistic essay quality
- **Feedback**: The system provides detailed feedback based on each score, offering suggestions for improvement.

## Technologies Used
- **Flask**: A lightweight Python web framework used to handle routing and server-side logic.
- **Jinja2**: A templating engine to dynamically generate HTML based on server-side data.
- **BERT**: A pre-trained model used for natural language processing tasks like grammar and lexical analysis.
- **Custom Functions**: Self-made functions for scoring essay quality, grammar, spelling, and word diversity.

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/automated-essay-grading.git
   cd automated-essay-grading
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app**:
   ```bash
   flask run
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000` to view the application.

## Usage
- Enter your essay in the provided text box and submit it.
- The system will process the essay and display scores and feedback based on the analysis from two different models.
- The results section will show predictions like grammar score, lexical score, organization scores, and overall essay quality.

