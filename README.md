# Hier
Machine learning translation of hieroglyphs

## hier2bin
BiLSTM  
input: a string of chars without spaces,  
output: array of 0 and 1 representing after which character there is a space  

## hier2hier
BiLSTM  
basic sequence to sequence, not working very well  

[## transfromer_01](transormer_01)
input: a string of chars without spaces,  
output: array of 0 and 1 representing after which character there is a space  
-> working

## transfromer_02
input: spaced text,  
output: transliteration  
