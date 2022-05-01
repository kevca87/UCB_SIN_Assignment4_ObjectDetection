window.addEventListener('load', (event) => {

    const baseUrl = "http://localhost:8080/objectdetection"
    
    function goTo(htmlFile){
        window.location.href = htmlFile;
    }

    function fetchPostFormConference(event)
    {
        //debugger;
        event.preventDefault();
        /*if(!event.currentTarget.dtName.value)
        {
            event.currentTarget.dtName.style.backgroundColor = 'red';
            return;
        }*/
        
        console.log('image: ')
        console.log(event.currentTarget.image.files[0])
        if (event.currentTarget.image.files[0] != undefined)
        {
            const formData = new FormData();
            formData.append('image', event.currentTarget.image.files[0]);
            //debugger;

            console.log(formData)

            let url = `${baseUrl}`;

            fetch(url, {
                method: 'POST',
                body: formData
            }).then(response => {
                if(response.status === 200){
                    //goTo('./events.html');
                } else {
                    response.text()
                    .then((error)=>{
                        alert(error);
                    });
                }
            });
        }
        else{
            console.log('no image');
        }
    }


    document.getElementById('upload-form').addEventListener('submit', fetchPostFormConference)

});