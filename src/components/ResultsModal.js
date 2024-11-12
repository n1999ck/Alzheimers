import React, { useState } from "react";
import Modal from "react-bootstrap/Modal"


export default function ResultsModal(props) {
    const [show, setShow] = useState(false);
    const handleClose = () => setShow(false);
    const handleShow = () => setShow(true);
  return (
    
    <Modal show={show} onHide={handleClose} classname={"ResultsModall"}>
        <Modal.Header closeButton={true}></Modal.Header>
        <Modal.Body>
            <Container fluid>
                <Row>
                    <Col>
                    <div>
                        <p>Hello</p>
                    </div>
                    </Col>
                </Row>
            </Container>
        </Modal.Body>
    </Modal>
  );
}
