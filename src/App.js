import React, { useState, useEffect } from "react";
import "./App.css";
import "bootstrap/dist/css/bootstrap.min.css";
import { useForm } from "react-hook-form";
import DemographicsForm from "./components/DemographicsForm";
import LifestyleAndBehaviorForm from "./components/LifestyleAndBehaviorForm";
import CognitiveFunctionalForm from "./components/CognitiveFunctionalForm";
import MedicalHistoryForm from "./components/MedicalHistoryForm";
import Header from "./components/ Header";

function App() {
  const { register, handleSubmit } = useForm();

  const onSubmit = (data) => {
    console.log(typeof data);
    console.log(data);
    console.log(JSON.stringify(data));
    console.log(Object.keys(data).length);
    fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Success:", data);
        alert(data);
      })
      .catch((error) => {
        console.error("Error:", error);
        alert("An error occurred while uploading the data. Please try again.");
      });
  };

  return (
    <div className="App">
      <Header />
      <div className="container-fluid mx-5">
        <form
          className="form"
          action="/api/upload"
          method="post"
          onSubmit={handleSubmit(onSubmit)}
        >
          <div className="row mx-5">
            <div className="col-lg-2 col">
              <h3 className="mt-4">Demographics and Personal Information</h3>

              <DemographicsForm register={register} />
            </div>

            <div className="col-lg-9 col">
              <div className="row mx-5">
                <div className=" row">
                  <h3 className="mt-4">Lifestyle and Behavior</h3>

                  <LifestyleAndBehaviorForm register={register} />
                </div>
                <div className="row">
                  <h3 className="mt-4">Medical History and Conditions</h3>

                  <MedicalHistoryForm register={register} />
                </div>
                <div className="row">
                  <h3 className="mt-4">Cognitive and Functional Assessments</h3>

                  <CognitiveFunctionalForm register={register} />
                </div>
              </div>
            </div>

            <button type="submit" className="btn btn-primary mt-4">
              Submit
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default App;
