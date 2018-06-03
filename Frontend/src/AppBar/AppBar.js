import React , {Component} from 'react';
import { withStyles } from '@material-ui/core/styles';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import Drawer from '@material-ui/core/Drawer';
import Button from '@material-ui/core/Button';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemIcon from '@material-ui/core/ListItemIcon';
import ListItemText from '@material-ui/core/ListItemText';

import IconButton from '@material-ui/core/IconButton';
import MenuIcon from '@material-ui/icons/Menu';
import DashboardIcon from '@material-ui/icons/Dashboard';
import SalesIcon from '@material-ui/icons/TrendingUp';
import RecommendationIcon from '@material-ui/icons/ThumbsUpDown';

const styles = {
    root: {
        flexGrow: 1,
    },
    flex: {
        flex: 1,
    },
    menuButton: {
        marginLeft: -12,
        marginRight: 20,
    },
};

class ButtonAppBar extends Component{
    state = {
        left: false
    };

    toggleDrawer = (side, open) => () => {
        this.setState({
            [side]: open,
        });
    };
    render(){
        const { classes , viewChange ,title} = this.props;
        return (
            <div className={classes.root}>
                <AppBar position="static">
                    <Toolbar>
                        <IconButton onClick={this.toggleDrawer('left', true)} className={classes.menuButton} color="inherit" aria-label="Menu">
                            <MenuIcon />
                        </IconButton>
                        <Typography variant="title" color="inherit" className={classes.flex}>
                            RecOMax - {title}
                        </Typography>
                    </Toolbar>
                </AppBar>
                <Drawer open={this.state.left} onClose={this.toggleDrawer('left', false)}>
                    <div
                        tabIndex={0}
                        role="button"
                        onClick={this.toggleDrawer('left', false)}
                        onKeyDown={this.toggleDrawer('left', false)}
                    >
                        <List component="nav" style={{minWidth:'300px'}}>
                            <ListItem button onClick={()=>viewChange('pred')}>
                                <ListItemIcon>
                                    <DashboardIcon/>
                                </ListItemIcon>
                                <ListItemText primary="Dashboard (Data Visualization)" />
                            </ListItem>
                            <ListItem button onClick={()=>viewChange('sales')}>
                                <ListItemIcon>
                                    <SalesIcon />
                                </ListItemIcon>
                                <ListItemText primary="Sales Algorithm (Notebook)" />
                            </ListItem>
                            <ListItem button onClick={()=>viewChange('recommend')}>
                                <ListItemIcon>
                                    <RecommendationIcon />
                                </ListItemIcon>
                                <ListItemText primary="Recommendation Algorithm (Notebook)" />
                            </ListItem>
                        </List>
                    </div>
                </Drawer>
            </div>)
    }

}

export default withStyles(styles)(ButtonAppBar);