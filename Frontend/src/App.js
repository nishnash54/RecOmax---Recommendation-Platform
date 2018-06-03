import React, {Component} from 'react';
import './App.css';

import AppBar from './AppBar/AppBar'
import Prediction from './Prediction/Prediction'
import Recommendation from './Recommendation/Recommendation'
import Sales from './Sales/Sales'

class App extends Component {
    constructor(props){
        super(props);
        this.state={
            pred:true,
            sales:false,
            recommend:false,
            title:'Aggregate Dashboard'
        };
        this.changeView=this.changeView.bind(this);
    }
    changeView(view){
        let tempState={};
        const appbarTitle={
            pred:'Aggregate Dashboard',
            sales:'Sales prediction Algorithm',
            recommend:'Recommendation Algorithm'
        };
        for(let i in this.state){
            (i==view)?tempState[i]=true:tempState[i]=false;
            if(i=='title')tempState[i]=appbarTitle[view];
        }
        this.setState(tempState);
    }
    render() {

        return (
            <div className="App">
                <AppBar title={this.state.title} viewChange={this.changeView}/>
                {
                    (this.state.pred)?
                    <Prediction/>   :''
                }
                {
                    (this.state.sales)?
                    <Sales/> :''
                }
                {
                    (this.state.recommend)?
                    <Recommendation/> :''
                }
            </div>
        );
    }
}

export default App;
