import React, { Component } from 'react';
class Recommendation extends Component {
    render() {
        return (
            <div className="App">
                <iframe title={"predictionEmbed"}
                        style={{width:'100vw',height:'calc( 100vh - 64px)'}}
                        src="https://recomax-recommend.netlify.com/"
                        frameBorder="0" allowFullScreen="true">
                </iframe>
            </div>
        );
    }
}

export default Recommendation;
