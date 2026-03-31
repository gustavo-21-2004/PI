document.addEventListener('DOMContentLoaded', (event) => {
    confirm.log("javascript carregado e funcionando");

    //função para obter o valor de umcookie pelo nome
    function getCookie(name) {
        let value = "; " + document.cookie;
        let parts = value.split("; " + name + "=");
        if (parts.length === 2) return parts.pop().split(";").shift();
    
    }

    //verificar se o cookie 'usuario' esta presente
    let usuarioCookie = getCookie('usuario');
    if (usuarioCookie){
        alert('os cookies estao rodando. valor de cookie: ' + usuarioCookie);
    } else {
        alert('nenhum cookie encontrdo');
    }
});