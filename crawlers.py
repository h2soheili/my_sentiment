
""""



var link = "https://www.ime.co.ir/subsystems/ime/auction/auction.ashx?fr=false&f=1403/4/11&t=1403/05/11&m=0&c=0&s=0&p=0&lang=8&order=asc&limit=200&offset="

var t = ""

async function get(){
    var offset = 0
    while (true){
      var res = await fetch(link + offset).then((res)=>res.json());
      var d = (res["rows"] || []);
      if (d.length == 0){
        console.log(offset)
        break
      }
      d.forEach((r)=>{
        Object.keys(r).forEach((k)=>{
            t+=" \n"+r[k]
        ""})
      })
      offset +=200
    }
    console.log(t)
}


get()

"""