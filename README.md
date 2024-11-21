# Predicting Alzheimer's

This is a simple program which takes data pertaining to an individual and feeds it to models to determine if they are at risk of developing Alzheimer's.

## Description

An in-depth paragraph about your project and overview of use.
The program takes in all kinds of data, including medical history, family history and traits that will be used by four trained deep learning models to predict if the user is at risk of developing Alzheimers. The models used are an FNN, an MLP, an SVM and a DF Classifier. They have each been trained using various features which now lead to an accurate prediction.

## Getting Started

### Dependencies
Necessary libraries are:
* PyTorch
* NumPy
* SciKit-Learn
* Pandas
* MatPlotLib

Operating system used for testing and development was conducted on Windows 10/11 and Ubuntu 22.04 (Pop!_OS).

### Installing

* Download the main branch from [GitHub](https://github.com/n1999ck/Alzheimers) and extract the files onto your machine. 

### Executing program

* How to run the program:
* Locate where your extracted files are saved
* Open the extracted file folder
* Open 'main.py'
* Press the run button in your IDE

## Authors

Contributors names  
'The Braingineers'  
**Kelly Payne**  
**Jaden Jefferson**  
**Jacob Tillmon**  
**Nick Miller**  

## Version History
* 0.1
    * Initial Release

## Acknowledgments
The Braingineers team would like to acknowledge the help of Dr. Yi Zhou in gathering resources to make this project possible.  
We would also like to thank Joe Papa, author of 'PyTorch Pocket Reference', which was heavily referenced in the creation of these models.  
In addition, Sebastion Raschka, Vahid Mirjalili and Yuxi (Hayden) Liu for their work in the book 'Machine Learning with PyTorch and SciKit-Learn'.


# MindWise: Web Client for Alzheimer's Predictor

## Description

This web application predicts the likelihood of Alzheimer’s disease based on user inputs in demographic, lifestyle, medical, and cognitive assessment categories. The frontend is built in React, while a Flask-based API processes the predictions using a machine learning model. The app aims to provide quick, accessible insights into potential Alzheimer's risk for personal or research purposes, differentiating itself by focusing on user-friendly form design and providing immediate prediction results.

## Badges

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-GNU%20GPL%20v3.0-blue)

## Visuals

![Screenshot of Application Form](https://github.com/n1999ck/Alzheimers/blob/webClient/screenshot.PNG?raw=true) 
*Form Screenshot: Users enter personal, lifestyle, and medical information for analysis.*

## Installation

### Requirements

To run this project, ensure you have the following software and packages installed:

#### System Requirements

- **Node.js**: Version 10.8.2 or higher TODO: versions
- **Python**: Version 3.12.1 or higher (compatible with `pip`)

#### Node Packages

These are managed via npm. Run the command below in the project root to install the necessary Node packages:

```bash
npm install
```

Alternatively, here is the main list of required packages, located in `package.json`:

- **react**: `18.3.1`
- **react-bootstrap**: `2.10.5` TODO: version
- **react-hook-form**: `7.53.1`
- **bootstrap**: `5.3.3`
- **flask**: `0.2.10`

#### Python (pip) Packages

Install the required Python packages by running:

```bash
pip install -r requirements.txt --TODO: make requirements.txt
```

The required packages, listed in `requirements.txt`, include:

torchsummary scikit-learn matplotlib pandas numpy flask python-dotenv

- **Flask**: `3.0.3`
- **numpy**: `2.1.1`
- **scikit-learn**: `1.5.2`
- **pandas**: `2.2.3`
- **torch**: `2.4.1`
- **torchsummary**: `1.5.1`


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

- **Form Input**: Enter patient data in each field. All input is done through radio buttons, numerical inputs, or selection drop-downs.
- **Submit**: Click the Submit button to send data to the back-end API for processing.
- **Output**: The app displays output in a modal.

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
```

## Authors and Acknowledgments

Developed by team Braingineers: Jaden Jefferson, Nick Miller, Kelly Payne, and Jacob Tillmon. Special thanks to Dr. Yi Zhou for guidance over the course of the project. Developed in partial fulfillment of the requirements for CPSC 4175: Software Engineering, Columbus State University.

## License

This project is licensed under the GNU General Public License v3.0.

## Project Status

This project will not be actively maintained after December 2024. However, it will be adapted into a new software in Spring 2025.
