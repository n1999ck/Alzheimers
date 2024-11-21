import React from "react";

export default function Header() {

    return (
        <nav className="navbar navbar-expand-lg" style={{backgroundColor: "#3C5087" }}>
            <div className="container-fluid" style={{backgroundColor: "#3C5087"}}>
                <a className="navbar-brand text-white" href="/">
                <img src="/assets/logo.svg" alt="" className="img me-4" height="70"/>
                Alzheimer's Disease Predictor</a>
            </div>
        </nav>
    );
}