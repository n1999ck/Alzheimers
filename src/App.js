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
// 1. Fix submit button - make smaller, sticky at bottom right
// 2. Fix styling to match medicalHistoryForm
// 3. Add validation perhaps
// 4. and/or change reaction to invalid inputs
// 5. Fix number inputs for decimals lol
// 6. Loading icon while waiting for model to start up

function App() {
  const { register, handleSubmit } = useForm();
  const [show, setShow] = useState(false);
  const handleClose = () => setShow(false);
  const handleShow = () => setShow(true);
  const [results, setResults] = useState({});

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
        handleShow();
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
            <div className="col-lg-2 col-sm-2">
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
            <Button type="submit" variant="success" size="lg" className="fixedButton">
              Submit
            </Button>
            
          </div>
        </form>
      </div>
    
    
    <Modal show={show} onHide={handleClose} classname={"ResultsModall"}>
        <Modal.Header closeButton={true}></Modal.Header>
        <Modal.Body>
            <Container fluid>
                <Row>
                    <Col>
                    <div>
                        <p>Hello</p>
                        <p>{results}</p>
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
