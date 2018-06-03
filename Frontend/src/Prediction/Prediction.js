import React, { Component } from 'react';
class Prediction extends Component {
    render() {
        return (
            <div className="App">
                <iframe title={"predictionEmbed"}
                        style={{width:'100vw',height:'calc( 100vh - 64px)'}}
                        src="https://app.powerbi.com/view?r=eyJrIjoiMTZhOWIyMTktM2MyNi00ZWFhLWJhY2EtMjZhZWNiNDhhNjkwIiwidCI6ImYyMDljNjUwLThkYzMtNDNiNy1hNTNiLWUxODhmYWFjM2UzZCJ9"
                        frameBorder="0" allowFullScreen="true">
                </iframe>
            </div>
        );
    }
}

export default Prediction;
