
# Multi-Modal Personalized Healthcare Agent

![Logo](./assets/logo-round.png)

## Overview

The **Multi-Modal Personalized Healthcare Agent** is an advanced AI-powered assistant designed to support healthcare professionals and patients. It enables seamless interaction through text, image, and speech, providing insights for diagnosis, treatment, and medical information retrieval.

---

## Features

- **Conversational AI**: Natural language chat for medical questions and advice.
- **Medical Image Analysis**: Upload images (e.g., skin lesions, X-rays, MRIs) for AI-powered analysis.
- **Speech-to-Text & Text-to-Speech**: Voice input and AI-generated speech responses using ElevenLabs.
- **Human Validation**: Supports human-in-the-loop validation for critical outputs.
- **Session Management**: Secure, cookie-based session tracking.
- **Modern UI**: Responsive, Kimi-inspired interface with dark mode.

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/multimodal-healthcare-agent.git
cd multimodal-healthcare-agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

- Copy `config.example.py` to `config.py` and set your API keys and settings.

### 4. Run the Application

```bash
python main.py
```

The app will be available at [http://localhost:8000](http://localhost:8000).

---

## API Endpoints

| Endpoint                | Method | Description                                 |
|-------------------------|--------|---------------------------------------------|
| `/`                     | GET    | Main web interface                          |
| `/chat`                 | POST   | Conversational AI (text queries)            |
| `/upload`               | POST   | Upload medical images for analysis          |
| `/validate`             | POST   | Human validation of AI outputs              |
| `/transcribe`           | POST   | Speech-to-text transcription                |
| `/generate-speech`      | POST   | Text-to-speech audio generation             |
| `/health`               | GET    | Health check for deployment                 |
| `/check-assets`         | GET    | Verify required assets exist                |

---

## Usage

- **Text Chat**: Type your medical question and receive instant AI responses.
- **Image Upload**: Click the image icon to upload medical images for analysis.
- **Voice Input**: Use the microphone icon to speak your query.
- **Dark Mode**: Toggle dark/light mode for comfortable viewing.

---

## Technologies

- **FastAPI**: Backend API framework
- **Jinja2**: HTML templating
- **Pydantic**: Data validation
- **ElevenLabs**: Speech synthesis and transcription
- **pydub**: Audio processing
- **Modern JS/CSS**: Responsive, accessible frontend

---

## Screenshots

![Screenshot](assets/demo_screenshot.png)

---

## Contributing

1. Fork the repo
2. Create a feature branch
3. Submit a pull request

---

## License

[MIT License](LICENSE)

---

## Disclaimer

This project is for research and educational purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment.

---

**Made with ❤️ for better healthcare.**