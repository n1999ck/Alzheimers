import React, { useState, useEffect } from 'react';
import './App.css';
import "halfmoon/css/halfmoon.min.css";

function App() {
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    fetch('/api/time').then(res => res.json()).then(data => {
      setCurrentTime(data.time);
    });
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <p>The current time is {currentTime}.</p>
          <form className='form' action='/api/upload' method='post' encType='multipart/form-data'>
      
            <h3>1. Demographics and Personal Information</h3>
            <label>PatientID:</label>
            <input type="number" name="patientID" required /><br />

            <label>Age:</label>
            <input type="number" name="age" min="0" required /><br />

            <label>Gender:</label>
            <select name="gender" required>
                <option value="0">Female</option>
                <option value="1">Male</option>
            </select><br />

            <label>Ethnicity:</label>
            <select name="ethnicity" required>
                <option value="0">Ethnicity 0</option>
                <option value="1">Ethnicity 1</option>
                <option value="2">Ethnicity 2</option>
                <option value="3">Ethnicity 3</option>
            </select><br />

            <label>Education Level:</label>
            <select name="educationLevel" required>
                <option value="0">Level 0</option>
                <option value="1">Level 1</option>
                <option value="2">Level 2</option>
                <option value="3">Level 3</option>
            </select><br />

            <h3>2. Lifestyle and Behavior</h3>
            <label>BMI:</label>
            <input type="number" name="bmi" min="15" max="40" step="0.1" required /><br />

            <label>Smoking:</label>
            <select name="smoking" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select><br />

            <label>Alcohol Consumption:</label>
            <input type="number" name="alcoholConsumption" min="0" max="20" step="0.1" required /><br />

            <label>Physical Activity:</label>
            <input type="number" name="physicalActivity" min="0" max="10" step="0.1" required /><br />

            <label>Diet Quality:</label>
            <input type="number" name="dietQuality" min="0" max="10" step="0.1" required /><br />

            <label>Sleep Quality:</label>
            <input type="number" name="sleepQuality" min="0" max="10" step="0.1" required /><br />

            <h3>3. Medical History and Conditions</h3>
            <label>Family History of Alzheimer's:</label>
            <select name="familyHistoryAlzheimers" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select><br />

            <label>Cardiovascular Disease:</label>
            <select name="cardiovascularDisease" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select><br />

            <label>Diabetes:</label>
            <select name="diabetes" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select><br />

            <label>Depression:</label>
            <select name="depression" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select><br />

            <label>Head Injury:</label>
            <select name="headInjury" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select><br />

            <label>Hypertension:</label>
            <select name="hypertension" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select><br />

            <label>Systolic BP:</label>
            <input type="number" name="systolicBP" min="90" max="179" required /><br />

            <label>Diastolic BP:</label>
            <input type="number" name="diastolicBP" min="60" max="119" required/><br />

            <label>Cholesterol Total:</label>
            <input type="number" name="cholesterolTotal" min="150" max="300" step="0.1" required/><br />

            <label>Cholesterol LDL:</label>
            <input type="number" name="cholesterolLDL" min="50" max="200" step="0.1" required/><br />

            <label>Cholesterol HDL:</label>
            <input type="number" name="cholesterolHDL" min="20" max="100" step="0.1" required/><br />

            <label>Cholesterol Triglycerides:</label>
            <input type="number" name="cholesterolTriglycerides" min="50" max="400" step="0.1" required/><br />

            <h3>4. Cognitive and Functional Assessments</h3>
            <label>MMSE:</label>
            <input type="number" name="mmse" min="0" max="30" step="0.1" required/><br />

            <label>Functional Assessment:</label>
            <input type="number" name="functionalAssessment" min="0" max="10" step="0.1" required/><br />

            <label>Memory Complaints:</label>
            <select name="memoryComplaints" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select><br />

            <label>Behavioral Problems:</label>
            <select name="behavioralProblems" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select><br />

            <label>ADL:</label>
            <input type="number" name="adl" min="0" max="10" step="0.1" required /><br />

            <label>Confusion:</label>
            <select name="confusion" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select><br />

            <label>Disorientation:</label>
            <select name="disorientation" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select><br />

            <label>Personality Changes:</label>
            <select name="personalityChanges" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select><br />

            <label>Difficulty Completing Tasks:</label>
            <select name="difficultyCompletingTasks" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select><br />

            <label>Forgetfulness:</label>
            <select name="forgetfulness" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select><br />

            <input type="submit" value="Submit" />

          </form>
      </header>
    </div>
  );
}

export default App;