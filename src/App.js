import React, { useState, useEffect } from 'react';
import './App.css';
import "halfmoon/css/halfmoon.min.css";
import {useForm} from 'react-hook-form';
function App() {  
  const { register, handleSubmit } = useForm();
  
  const onSubmit = (data) => {
    console.log(typeof data);
    console.log(data);
    console.log(JSON.stringify(data));
    console.log(Object.keys(data).length);
    fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(data => {
      console.log('Success:', data);
      alert("Data has been uploaded successfully!");
    })
    .catch((error) => {
      console.error('Error:', error);
      alert("An error occurred while uploading the data. Please try again.");
    });
  };

  return (
    <div className="App container">
      <header className="App-header">
        
        <form className="form" action="/api/upload" method="post" onSubmit={handleSubmit(onSubmit)}>
          <h3 className="mt-4">Demographics and Personal Information</h3>

          
          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">PatientID:</label>
            <div className="col-sm-9">
              <input type="number" className="form-control" {...register("patientID", { required: true })} />
              <div className='form-text'>Please enter a unique patient ID.</div>
            </div>
          </div>

          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Age:</label>
            <div className="col-sm-9">
              <input type="number" className="form-control" {...register("age", { min: 0, required: true })} />
              <div className='form-text'>Enter a whole number above 0.</div>
            </div>
          </div>

          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Gender:</label>
            <div className="col-sm-9">
              <select className="form-select" {...register("gender", { required: true })}>
                <option value="0">Female</option>
                <option value="1">Male</option>
              </select>
            </div>
          </div>

          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Ethnicity:</label>
            <div className="col-sm-9">
              <select className="form-select" {...register("ethnicity", { required: true })}>
                <option value="0">Hispanic or Latino</option>
                <option value="1">White (Non-Hispanic)</option>
                <option value="2">Black or African American</option>
                <option value="3">Asian</option>
              </select>
            </div>
          </div>

          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Education Level:</label>
            <div className="col-sm-9">
              <select className="form-select" {...register("educationLevel", { required: true })}>
                <option value="0">Less than High School</option>
                <option value="1">High School Diploma or GED</option>
                <option value="2">Some College or Associate's Degree</option>
                <option value="3">Bachelor’s Degree or Higher</option>
              </select>
            </div>
          </div>

          <h3 className="mt-4">Lifestyle and Behavior</h3>

          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">BMI:</label>
            <div className="col-sm-9">
              <input type="number" className="form-control" {...register("bmi", { min: 15, max: 50, step: 0.1, required: true })} />
              <div className='form-text'>Enter a number between 15 and 60.</div>
            </div>
          </div>

          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Smoking:</label>
            <div className="col-sm-9">
              <select className="form-select" {...register("smoking", { required: true })}>
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
            </div>
          </div>

          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Alcohol Consumption:</label>
            <div className="col-sm-9">
              <input type="number" className="form-control" {...register("alcoholConsumption", { min: 0, max: 20, step: 0.1, required: true })} />
              <div className='form-text'>Enter a floating point number between 0 and 20.</div>
            </div>
          </div>

          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Physical Activity:</label>
            <div className="col-sm-9">
            <input type="number" className="form-control" {...register("physicalActivity", { min: 0, max: 10, step: 0.1, required: true })} />
            <div className='form-text'>Enter a floating point number between 0 and 10.</div>
            </div>
          </div>

          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Diet Quality:</label>
            <div className="col-sm-9">
            <input type="number" className="form-control" {...register("dietQuality", { min: 0, max: 10, step: 0.1, required: true })} />
            <div className='form-text'>Enter a floating point number between 0 and 10.</div>
            </div>
          </div>

          <div className="row mb-3 mb-sm-3">
            <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Sleep Quality:</label>
            <div className="col-sm-9">
            <input type="number" className="form-control" {...register("sleepQuality", { min: 0, max: 10, step: 0.1, required: true })} />
            <div className='form-text'>Enter a floating point number between 0 and 10.</div>
            </div>
          </div>

          <h3 className="mt-4">Medical History and Conditions</h3>
          
          <div className="row mb-3 mb-sm-3">
                <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Family History of Alzheimer's:</label>
                <div className="col-sm-9 form-check form-check-inline">
            
                    <input type="radio" className='form-check-input' value="0" {...register("familyHistoryAlzheimers", { required: true })} />
                    <label className='form-check-label'>Yes</label>
                
                    <input type="radio" className='form-check-input' value="1" {...register("familyHistoryAlzheimers", { required: true })} />
                    <label className='form-check-label'>No</label>
                
                </div>
            </div>

            <div className="row mb-3 mb-sm-3">
                <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Cardiovascular Disease:</label>
                <div className="col-sm-9">
                <select name="cardiovascularDisease" className="form-select" required>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
                </div>
            </div>

            <div className="row mb-3 mb-sm-3">
                <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Diabetes:</label>
                <div className="col-sm-9">
                <select name="diabetes" className="form-select" required>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
                </div>
            </div>

            <div className="row mb-3 mb-sm-3">
                <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Depression:</label>
                <div className="col-sm-9">
                <select name="depression" className="form-select" required>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
                </div>
            </div>

            <div className="row mb-3 mb-sm-3">
    <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Head Injury:</label>
    <div className="col-sm-9">
        <select name="headInjury" className="form-select" required>
            <option value="0">No</option>
            <option value="1">Yes</option>
        </select>
    </div>
</div>

<div className="row mb-3 mb-sm-3">
    <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Hypertension:</label>
    <div className="col-sm-9">
        <select name="hypertension" className="form-select" required>
            <option value="0">No</option>
            <option value="1">Yes</option>
        </select>
    </div>
</div>

<div className="row mb-3 mb-sm-3">
    <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Systolic BP:</label>
    <div className="col-sm-9">
        <input type="number" name="systolicBP" min="90" max="179" className="form-control" required />
        <div className='form-text'>Enter an integer between 90 and 179.</div>
    </div>
</div>

<div className="row mb-3 mb-sm-3">
    <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Diastolic BP:</label>
    <div className="col-sm-9">
        <input type="number" name="diastolicBP" min="60" max="119" className="form-control" required />
        <div className='form-text'>Enter an integer between 60 and 119.</div>
    </div>
</div>

<div className="row mb-3 mb-sm-3">
    <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Cholesterol Total:</label>
    <div className="col-sm-9">
        <input type="number" name="cholesterolTotal" min="150" max="300" step="0.1" className="form-control" required />
        <div className='form-text'>Enter a floating point number between 150 and 300.</div>
    </div>
</div>

<div className="row mb-3 mb-sm-3">
    <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Cholesterol LDL:</label>
    <div className="col-sm-9">
        <input type="number" name="cholesterolLDL" min="50" max="200" step="0.1" className="form-control" required />
        <div className='form-text'>Enter a floating point number between 50 and 200.</div>
    </div>
</div>

<div className="row mb-3 mb-sm-3">
    <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Cholesterol HDL:</label>
    <div className="col-sm-9">
        <input type="number" name="cholesterolHDL" min="20" max="100" step="0.1" className="form-control" required />
        <div className='form-text'>Enter a floating point number between 20 and 100.</div>
    </div>
</div>

<div className="row mb-3 mb-sm-3">
    <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Cholesterol Triglycerides:</label>
    <div className="col-sm-9">
        <input type="number" name="cholesterolTriglycerides" min="50" max="400" step="0.1" className="form-control" required />
        <div className='form-text'>Enter a floating point number between 50 and 400.</div>
    </div>
</div>

<h3 className="mt-4">Cognitive and Functional Assessments</h3>

<div className="row mb-3 mb-sm-3">
    <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">MMSE:</label>
    <div className="col-sm-9">
        <input type="number" className="form-control" {...register("mmse", { min: 0, max: 30, step: 0.1, required: true })} />
        <div className='form-text'>Enter a floating point number between 0 and 30.</div>
    </div>
</div>

<div className="row mb-3 mb-sm-3">
    <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Functional Assessment:</label>
    <div className="col-sm-9">
        <input type="number" maxLength={80} className="form-control" {...register("functionalAssessment", { min: 0, max: 10, step: 0.1, required: true })} />
        <div className='form-text'>Enter a floating point number between 0 and 10.</div>
    </div>
</div>

<div className="row mb-3 mb-sm-3">
    <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Memory Complaints:</label>
    <div className="col-sm-9">
        <select className="form-select" {...register("memoryComplaints", { required: true })}>
            <option value="0">No</option>
            <option value="1">Yes</option>
        </select>
    </div>
</div>

<div className="row mb-3 mb-sm-3">
    <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Behavioral Problems:</label>
    <div className="col-sm-9">
        <select className="form-select" {...register("behavioralProblems", { required: true })}>
            <option value="0">No</option>
            <option value="1">Yes</option>
        </select>
    </div>
</div>

<div className="row mb-3 mb-sm-3">
    <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">ADL:</label>
    <div className="col-sm-9">
        <input type="number" className="form-control" {...register("adl", { min: 0, max: 10, step: 0.1, required: true })} />
        <div className='form-text'>Enter a floating point number between 0 and 10.</div>
    </div>
</div>

<div className="row mb-3 mb-sm-3">
    <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Confusion:</label>
    <div className="col-sm-9">
        <select className="form-select" {...register("confusion", { required: true })}>
            <option value="0">No</option>
            <option value="1">Yes</option>
        </select>
    </div>
</div>

<div className="row mb-3 mb-sm-3">
    <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Disorientation:</label>
    <div className="col-sm-9">
        <select className="form-select" {...register("disorientation", { required: true })}>
            <option value="0">No</option>
            <option value="1">Yes</option>
        </select>
    </div>
</div>

<div className="row mb-3 mb-sm-3">
    <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Personality Changes:</label>
    <div className="col-sm-9">
        <select className="form-select" {...register("personalityChanges", { required: true })}>
            <option value="0">No</option>
            <option value="1">Yes</option>
        </select>
    </div>
</div>

<div className="row mb-3 mb-sm-3">
    <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Difficulty Completing Tasks:</label>
    <div className="col-sm-9">
        <select className="form-select" {...register("difficultyCompletingTasks", { required: true })}>
            <option value="0">No</option>
            <option value="1">Yes</option>
        </select>
    </div>
</div>

<div className="row mb-3 mb-sm-3">
    <label className="col-sm-3 col-form-label col-form-label-lg mb-1 mb-sm-0">Forgetfulness:</label>
    <div className="col-sm-9">
        <select className="form-select" {...register("forgetfulness", { required: true })}>
            <option value="0">No</option>
            <option value="1">Yes</option>
        </select>
    </div>
</div>

          <button type="submit" className="btn btn-primary mt-4">Submit</button>
        </form>
      </header>
    </div>
  );
}

export default App;