<template>
   <div id="app">
     <h1>Transformer Model Prediction</h1>
     <input v-model="inputText" placeholder="Enter your text" />
     <button @click="makePrediction">Predict</button>
     <div v-if="output">
       <h2>Output:</h2>
       <p>{{ output }}</p>
     </div>
   </div>
 </template>
 
 <script>
 export default {
   data() {
     return {
       inputText: '',
       output: null,
     };
   },
   methods: {
     async makePrediction() {
       const response = await fetch('http://localhost:8000/predict', {
         method: 'POST',
         headers: {
           'Content-Type': 'application/json',
         },
         body: JSON.stringify({ input_text: this.inputText }),
       });
       const data = await response.json();
       this.output = data.output;
     },
   },
 };
 </script>
 
 <style>
 /* Add some basic styles */
 </style>
 