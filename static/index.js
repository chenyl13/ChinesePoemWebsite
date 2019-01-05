$(document).ready(function(){
    var length2 = document.getElementById('length2')
    var poem = document.getElementById('poem');
    var b5 = document.getElementById('5');
    var b7 = document.getElementById('7');
    var length = document.getElementById('length')
    var refresh = document.getElementById('refresh')
    if(length2.value == '5'){
        length.value = "5";
        b5.setAttribute('class', "ui big teal button");
    }
    else if(length2.value == '7'){
        length.value = "7";
        b7.setAttribute('class', "ui big teal button");
    }
    else{
        poem.style.display = "none";
    }
    function press5(){
        if(b5.className.match('active') == null){
            b5.setAttribute('class', "ui big teal button");
            b7.setAttribute('class', "ui big teal basic button");
            length.value = "5";
        }
    }
    function press7(){
        if(b7.className.match('active') == null){
            b7.setAttribute('class', "ui big teal button");
            b5.setAttribute('class', "ui big teal basic button");
            length.value = "7";
        }
    }
    function overr(){
        refresh.setAttribute('class', "ui right floated teal big icon button");
    }
    function outr(){
        refresh.setAttribute('class', "ui right floated teal basic big icon button");
    }
    b5.onclick = press5;
    b7.onclick = press7;
    refresh.onmouseover = overr;
    refresh.onmouseout = outr;
});