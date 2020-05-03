import React, { Component } from "react";
import {
    Alert,
    AppRegistry,
    Button,
    Image,
    StyleSheet,
    Text,
    View,
} from "react-native";
import { Dialog, ProgressDialog, ConfirmDialog } from "react-native-simple-dialogs";

const styles = StyleSheet.create({
    container: {
        alignItems: "center",
        backgroundColor: "#F5FCFF",
        flex: 1,
        justifyContent: "center",
    },
    welcomeText: {
        fontSize: 20,
        margin: 10,
        textAlign: "center",
    },
    exampleText: {
        fontSize: 20,
        marginBottom: 25,
        textAlign: "center",
    },
    instructionsText: {
        color: "#333333",
        fontSize: 16,
        marginBottom: 40,
        textAlign: "center",
    },
});

export default class App extends Component {
    state = {}

    openDialog = (show) => {
        this.setState({ showDialog: show });
    }

    openConfirm = (show) => {
        this.setState({ showConfirm: show });
    }


    optionYes = () => {
        this.openConfirm(false);
        // Yes, this is a workaround :(
        // Why? See this https://github.com/facebook/react-native/issues/10471
        setTimeout(
            () => {
                Alert.alert("gg");
            },
            300,
        );
    }

    optionNo = () => {
        this.openConfirm(false);
        // Yes, this is a workaround :(
        // Why? See this https://github.com/facebook/react-native/issues/10471
        setTimeout(
            () => {
                Alert.alert("gg");
            },
            300,
        );
    }

    render() {
        return (
            <View style={ styles.container }>

                <Text style={ styles.exampleText }>
                    Enter Home Information:
                </Text>
                <Button
                    onPress={ () => this.openConfirm(true) }
                    title="Lot Size"
                />

                <View style={ { height: 30 } } />

               <Button
                   onPress={ () => this.openDialog(true) }
                    title="Bedrooms"
                />
                <View style={ { height: 30 } } />

                <Button
                    onPress={ () => this.openDialog(true) }
                     title="Bathrooms"
                 />
                 <View style={ { height: 30 } } />

                 <Button
                   onPress={ () => this.openDialog(true) }
                    title="HouseStyle"
                 />

                  <View style={ { height: 30 } } />

               <Button
                    onPress={ () => this.openDialog(true) }
                    title="Neighborhood"
                 />

                <View style={ { height: 30 } } />

                <Button
                    onPress={ () => this.openDialog(true) }
                    title="Overall Quality"
                 />

                <View style={ { height: 30 } } />
                <Button
                    onPress={ () => this.openDialog(true) }
                    title="YearBuilt"
                 />

                <View style={ { height: 30 } } />

                <Button
                    onPress={ () => this.openDialog(true) }
                    title="Exterior Condition"
                 />

                <View style={ { height: 30 } } />

              <Button
                  onPress={ () => this.openDialog(true) }
                   title="Floors"
               />
               <View style={ { height: 30 } } />


                <Dialog
                    title="In progress"
                    animationType="fade"
                    contentStyle={
                        {
                            alignItems: "center",
                            justifyContent: "center",
                        }
                    }
                    onTouchOutside={ () => this.openDialog(false) }
                    visible={ this.state.showDialog }
                >

                    <Text style={ { marginVertical: 30 } }>
                        Welcome to this dialog box. Functionality curated to the specific edit profile option coming soon in sprint 3. :-)
                    </Text>
                    <Button
                        onPress={ () => this.openDialog(false) }
                        style={ { marginTop: 10 } }
                        title="CLOSE"
                    />
                </Dialog>

                <ConfirmDialog
                    title="lot size"
                    message="enter lot size bruv?"
                    onTouchOutside={ () => this.openConfirm(false) }
                    visible={ this.state.showConfirm }
                    negativeButton={
                        {
                            title: "NO",
                            onPress: this.optionNo,
                            // disabled: true,
                            titleStyle: {
                                color: "blue",
                                colorDisabled: "aqua",
                            },
                            style: {
                                backgroundColor: "transparent",
                                backgroundColorDisabled: "transparent",
                            },
                        }
                    }
                    positiveButton={
                        {
                            title: "YES",
                            onPress: this.optionYes,
                        }
                    }
                />

                <ProgressDialog
                    title="Progress Dialog"
                    activityIndicatorColor="blue"
                    activityIndicatorSize="large"
                    animationType="slide"
                    message="Please, wait..."
                    visible={ this.state.showProgress }
                />
            </View>
        );
    }
}

AppRegistry.registerComponent('Sample', () => App);
