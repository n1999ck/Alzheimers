import React, { useState, useEffect } from "react";
import "./App.css";
import  Button from "react-bootstrap/Button";
import { useForm } from "react-hook-form";
import DemographicsForm from "./components/DemographicsForm";
import LifestyleAndBehaviorForm from "./components/LifestyleAndBehaviorForm";
import CognitiveFunctionalForm from "./components/CognitiveFunctionalForm";
import MedicalHistoryForm from "./components/MedicalHistoryForm";
import Header from "./components/Header";
import Modal from "react-bootstrap/Modal";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";


//TODO:
// --------1. Fix submit button - make smaller, sticky at bottom right
// 2. Fix styling to match medicalHistoryForm
// 3. Add validation perhaps
// 4. and/or change reaction to invalid inputs
// 5. Fix number inputs for decimals lol!!
// 6. Loading icon while waiting for model to start up
// ------7. 3C5087 - color for navbar and titles
// ------8. Patient icon - male or female change
// 9. Change title sizes 
// 10. User Document
// 11. Finish modal
// 12. Later on - about page
// 13. Later on - save entries, cache result
// 14. Model architecture drawings ? 

// BEFORE PRESENTATION:
// 1. Fix styling
// 2. Change title sizes
// 3. Finish modal !!!
// 4. DECIMALS


function App() {
  const { register, handleSubmit } = useForm();
  const [show, setShow] = useState(false);
  const handleClose = () => setShow(false);
  const handleShow = () => setShow(true);
  const [results, setResults] = useState({});
  const [accuracies, setAccuracies] = useState({});
  const [diagnosis, setDiagnosis] = useState("");

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
        setResults(data);
        calculateDiagnosis();
      })
      .catch((error) => {
        console.error("Error:", error);
        alert("An error occurred while uploading the data. Please try again.");
      });
      fetch("/api/accuracies")
    .then((response) => response.json())
    .then((data) => {
      console.log("Success:", data);
      setAccuracies(data);
      calculateDiagnosis();
      handleShow();
    })
    .catch((error) => {
      console.error("Error:", error);
      alert("An error occurred while uploading the data. Please try again.");
    });
  };


  const calculateDiagnosis = () => {
    var weightedResults = [];
    for (let key in results) {
      if (results.hasOwnProperty(key) && accuracies.hasOwnProperty(key)) {
        console.log("Result: " + results[key]);
        console.log("Accuracy: " + accuracies[key]);
        weightedResults.push(results[key] * accuracies[key]);
      }
    }
    console.log("Weighted results: ", weightedResults);
    setDiagnosis(weightedResults.reduce((a, b) => a + b, 0));
  }

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
            <div className="col-lg-2 col-sm-2">
              <h3 className="mt-4 sectionTitle">Demographics and Personal Information</h3>

              <DemographicsForm register={register} />
            </div>

            <div className="col-lg-9 col">
              <div className="row mx-5">
                <div className=" row">
                  <h3 className="mt-4 sectionTitle">Lifestyle and Behavior</h3>

                  <LifestyleAndBehaviorForm register={register} />
                </div>
                <div className="row">
                  <h3 className="mt-4 sectionTitle">Medical History and Conditions</h3>

                  <MedicalHistoryForm register={register} />
                </div>
                <div className="row">
                  <h3 className="mt-4 sectionTitle">Cognitive and Functional Assessments</h3>

                  <CognitiveFunctionalForm register={register} />
                </div>
              </div>
            </div>
            <Button type="submit" onClick= {console.log(register)}variant="success" size="lg" className="fixedButton">
              Submit
            </Button>
            
          </div>
        </form>
      </div>
    
    
    <Modal show={show} onHide={handleClose} className={"ResultsModal"}>
        <Modal.Header closeButton={true}>
            <Modal.Title>Diagnosis: {diagnosis === 0 ? "Alzheimer's Unlikely" : "Alzheimer's Likely"}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
            <Container fluid>
                <Row>
                    <Col>
                    <div>
                      
                        <p>Our neural networks have determined that the patient {diagnosis === 0 ? "is unlikely to suffer from Alzheimer's disease." : "is likely to suffer from Alzheimer's disease."}</p>
                    </div>
                    </Col>
                </Row>
            </Container>
        </Modal.Body>
    </Modal>
    </div>
  );
}

export default App;
