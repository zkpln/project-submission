import React from 'react';
import { View, Button, TextInput, StyleSheet, TouchableOpacity, Text } from 'react-native';
import { KeyboardAwareScrollView } from 'react-native-keyboard-aware-scroll-view';

export default class SignUp extends React.Component {
  render() {
    //const { navigate } = this.props.navigation;
    return (
      <KeyboardAwareScrollView
        style={{ backgroundColor: '#fff' }}
        resetScrollToCoords={{ x: 0, y: 0 }}
        contentContainerStyle={styles.container}
        scrollEnabled={true}
      >

      <Text style={ styles.title3 }>
                    Your Results:

                </Text>
      <Text style={ styles.title }>
                    Your House is worth "$250,000"


                </Text>

                <Text style={ styles.title2 }>
                    Your most valuable feature is your "neighborhood"

                </Text>

                <Text style={ styles.title2 }>
                    Your least valuable feature is your "Year built"

                </Text>

      </KeyboardAwareScrollView>
    )
  }
}
const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center'
  },

  title: {
    width: 350,
    height: 55,
    backgroundColor: 'black',
    margin: 10,
    padding: 15,
    color: 'pink',
    borderRadius: 30,
    fontSize: 18,
    fontWeight: '500',
  },
  title2: {
    width: 350,
    height: 55,
    backgroundColor: 'light blue',
    margin: 10,
    padding: 15,
    color: 'red',
    borderRadius: 10,
    fontSize: 13,
    fontWeight: '500',
  },
  title3: {
    width: 350,
    height: 60,
    backgroundColor: 'blue',
    margin: 10,
    padding: 20,
    color: 'red',
    borderRadius: 10,
    fontSize: 20,
    fontWeight: '500',
  },
})
