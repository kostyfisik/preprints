#version 3.7;
global_settings { assumed_gamma 1.2 } 

#include "colors.inc"
#include "textures.inc"
#include "metals.inc"
#include "glass.inc"
//-------------------------------------------------------------

camera {
   location <100, 40, 50>
   angle 15 
   right x*image_width/image_height
   look_at <0,0,0>
}

light_source {<200, 50, 100> color White  fade_distance 200 fade_power 0 }
background { White }
//-------------------------------------------------------------

difference {
  sphere {
    <0, 0, 0>, 10
    pigment{color Black} interior { I_Glass caustics 1 } finish { reflection 0.15 ambient 0.5 diffuse 0.1 specular 0.7 roughness .4 }  
    }

  box { <0, -15, 0>, <15, 15, 15>  
    pigment{color Black} interior { I_Glass caustics 1 } finish { reflection 0.01 ambient 0.5 diffuse 0.1 specular 0.7 roughness .4 }  
    }
  }
     
difference {
  sphere {
    <0, 0, 0>, 7
    pigment{color Gold} interior { I_Glass caustics 1 } finish { reflection 0.1 ambient 0.5 diffuse 0.5 specular 0.1 roughness .2 }  
    }

  box { <0.1, 0, 0.1>, <15, 15, 15> 
    pigment{color Gold} interior { I_Glass caustics 1 } finish { reflection 0.1 ambient 0.5 diffuse 0.5 specular 0.1 roughness .2 }
    }
   }
              
              
sphere {
  <0, 0, 0>, 5.5
  pigment{color Black} interior { I_Glass caustics 1 } finish { reflection 0.01 ambient 0.5 diffuse 0.1 specular 0.7 roughness .4 }  
  }        
//-------------------------------------------------------------
                 
                 
text{ttf "LiberationSans-Regular.ttf", "Ag",1, 0 
     texture{ pigment{ color Gold}
              finish { phong 1 reflection 0.2}}
     rotate<-18,205,0> scale<3,3,3> translate<5, -3 ,5> }    
                      
text{ttf "LiberationSans-Regular.ttf", "Si",1, 0 
     texture{ pigment{ color Black}
              finish { phong 1 reflection 0.2}}
     rotate<15, 205,0> scale<3,3,3> translate<3.6, 0.8 ,4.7> }    
     
text{ttf "LiberationSans-Regular.ttf", "Si",1, 0 
     texture{ pigment{ color Black}
              finish { phong 1 reflection 0.2}}
     rotate<7,280,0> scale<3,3,3> translate<10, 1 ,-3> }    