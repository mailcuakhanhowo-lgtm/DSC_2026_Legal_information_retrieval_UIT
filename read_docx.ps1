Expand-Archive -Path 'DSC2026_Task1_LegalIR_Data_Overview.docx' -DestinationPath 'temp_docx' -Force
$xml = [xml](Get-Content 'temp_docx\word\document.xml' -Raw)
$ns = @{ w = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main' }
Select-Xml -Xml $xml -XPath '//w:p' -Namespace $ns | ForEach-Object {
    $text = (Select-Xml -Xml $_.Node -XPath './/w:t' -Namespace $ns | ForEach-Object { $_.Node.InnerXML }) -join ''
    if ($text) { Write-Output $text }
}
Remove-Item -Recurse -Force 'temp_docx'
