# Alzheimer’s Disease Prediction App

## Description

This web application predicts the likelihood of Alzheimer’s disease based on user inputs in demographic, lifestyle, medical, and cognitive assessment categories. The frontend is built in React, while a Flask-based API processes the predictions using a machine learning model. The app aims to provide quick, accessible insights into potential Alzheimer's risk for personal or research purposes, differentiating itself by focusing on user-friendly form design and providing immediate prediction results.

## Badges

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-GNU%20GPL%20v3.0-blue)

## Visuals

![Screenshot of Application Form](https://via.placeholder.com/500x300)  TODO: screenshot
*Form Screenshot: Users enter personal, lifestyle, and medical information for analysis.*

## Installation

### Requirements

To run this project, ensure you have the following software and packages installed:

#### System Requirements

- **Node.js**: Version X.X.X or higher TODO: versions
- **Python**: Version X.X.X or higher (compatible with `pip`)

#### Node Packages

These are managed via npm. Run the command below in the project root to install the necessary Node packages:

```bash
npm install
```

Alternatively, here is the main list of required packages, located in `package.json`:

- **react**: `18.3.1`
- **react-bootstrap**: `X.X.X` TODO: version
- **react-hook-form**: `7.53.1`
- **react-bootstrap**: `5.3.3`
- **halfmoon**: `2.0.2`
- **flask**: `0.2.10`

#### Python (pip) Packages

Install the required Python packages by running:

```bash
pip install -r requirements.txt --TODO: make requirements.txt
```

The required packages, listed in `requirements.txt`, include:

- **Flask**: `X.X.X`
- **numpy**: `X.X.X`
- **scikit-learn**: `X.X.X`
- TODO: finish this lol

---

This way, users know where to look and how to install each package group. Let me know if you'd like any adjustments!

### Steps

1. **Clone the repository**:

   ```bash
   git clone https://github.com/n1999ck/Alzheimers-Prediction-App.git
   cd Alzheimers-Prediction-App
   ```

2. **Backend setup**:
   - Create a virtual environment:

     ```bash
     python3 -m venv venv
     source venv/bin/activate   # On Windows use `venv\Scripts\activate`
     ```

   - Install Flask and other dependencies:

     ```bash
     pip install -r requirements.txt
     ```

3. **Frontend setup**:
   - Navigate to the `client` directory and install npm dependencies:

     ```bash
     cd client
     npm install
     ```

4. **Run the application**:
   - Start the Flask server:

     ```bash
     yarn start-api
     ```

   - In a new terminal window, start the React development server:

     ```bash
     yarn start
     ```

   The app will be accessible at `http://localhost:3000`.

## Usage

- **Form Input**: Users fill out sections on demographics, lifestyle, medical history, and cognitive assessments.
- **Submit**: Click the Submit button to send data to the back-end API.
- **Output**: The app returns a prediction based on the user’s input data.

## Support

For questions, please open an issue in the GitHub repository.

## Roadmap

- **Upcoming Features**:
  - Improved prediction accuracy updates to decision algorithm.
  - Enhanced visualization of prediction results.

## Contributing

We welcome contributions! To get started:

1. Fork this repository.
2. Make your changes in a new branch.
3. Submit a pull request with a clear description of the changes.

For testing, use:

```bash
npx cypress open # Frontend tests
pytest     # Backend tests
```

## Authors and Acknowledgments

Developed by team Braingineers: Jaden Jefferson, Nick Miller, Kelly Payne, and Jacob Tillmon. Special thanks to Dr. Yi Zhou for guidance over the course of the project. Developed in partial fulfillment of the requirements for CPSC 4175: Software Engineering, Columbus State University.

## License

This project is licensed under the GNU General Public License v3.0.

## Project Status

This project will not be actively maintained after December 2024. However, it will be adapted into a new software in Spring 2025.
